import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random

# Import custom modules
from loader import (
    load_dti_dataset, process_dti_data, create_data_loaders, 
    custom_collate_fn, move_batch_to_device
)
from models import CATDTIpy
from loss import DTILoss, calculate_metrics, evaluate_model
from args import get_args, save_args

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        batch = move_batch_to_device(batch, device)
        labels = batch['labels'].float()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store results
        total_loss += loss.item()
        all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            labels = batch['labels'].float()
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            # Store results
            total_loss += loss.item()
            all_predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train Acc')
    axes[0, 1].plot(history['val_accuracy'], label='Val Acc')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    
    # AUC
    axes[1, 0].plot(history['train_auc'], label='Train AUC')
    axes[1, 0].plot(history['val_auc'], label='Val AUC')
    axes[1, 0].set_title('AUC')
    axes[1, 0].legend()
    
    # F1
    axes[1, 1].plot(history['train_f1'], label='Train F1')
    axes[1, 1].plot(history['val_f1'], label='Val F1')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def main():
    # Get arguments
    args = get_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Save arguments
    save_args(args, os.path.join(args.save_dir, 'args.json'))
    
    # Load and process data
    print("Loading dataset...")
    df = load_dti_dataset(args.dataset_path, args.max_samples)
    
    print("Processing data...")
    processed_data = process_dti_data(df, max_samples=args.max_samples)
    
    print("Creating data loaders...")
    train_data, val_data, test_data = create_data_loaders(
        processed_data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=custom_collate_fn,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=custom_collate_fn,
                           num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=custom_collate_fn,
                            num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    # Initialize model with arguments
    model = CATDTIpy(
        drug_node_feat_dim=args.drug_node_feat_dim,
        drug_embedding=args.drug_embedding,
        drug_hidden_feats=args.drug_hidden_feats,
        protein_vocab_size=args.protein_vocab_size,
        protein_emb_dim=args.protein_emb_dim,
        protein_max_len=args.protein_max_len,
        protein_num_layers=args.protein_num_layers,
        protein_num_attention_heads=args.protein_num_attention_heads,
        protein_ff_expansion=args.protein_ff_expansion,
        protein_ff_dropout=args.protein_ff_dropout,
        protein_attn_dropout=args.protein_attn_dropout,
        protein_conv_dropout=args.protein_conv_dropout,
        protein_conv_kernel=args.protein_conv_kernel,
        mlp_in_dim=args.mlp_in_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_out_dim=args.mlp_out_dim
    ).to(device)
    
    # Initialize loss function
    criterion = DTILoss(loss_type=args.loss_type, alpha=args.focal_alpha, beta=args.focal_gamma)
    
    # Initialize optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                             nesterov=args.nesterov, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': []
    }
    
    # Initialize learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                             gamma=args.scheduler_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                        patience=args.scheduler_patience)
    else:
        scheduler = None
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.3f}, "
              f"AUC: {train_metrics['auc']:.3f}, "
              f"F1: {train_metrics['f1']:.3f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.3f}, "
              f"AUC: {val_metrics['auc']:.3f}, "
              f"F1: {val_metrics['f1']:.3f}")
        
        # Update history
        for key in history.keys():
            if key.startswith('train_'):
                metric_name = key[6:]  # Remove 'train_' prefix
                history[key].append(train_metrics[metric_name])
            elif key.startswith('val_'):
                metric_name = key[4:]   # Remove 'val_' prefix
                history[key].append(val_metrics[metric_name])
        
        # Save best model
        if val_metrics['auc'] > best_val_auc + args.early_stopping_min_delta:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, 
                          os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved! Val AUC: {best_val_auc:.3f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement")
            break
        
        # Save checkpoint at intervals
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics,
                          os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics, test_predictions, test_targets = evaluate_model(
        model, test_loader, device, criterion
    )
    
    print(f"Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"AUC: {test_metrics['auc']:.3f}")
    print(f"F1: {test_metrics['f1']:.3f}")
    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['recall']:.3f}")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'training_history': history,
        'args': vars(args)
    }
    
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))
    
    print(f"\nTraining completed! Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
