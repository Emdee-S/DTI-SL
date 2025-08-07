#!/usr/bin/env python3
"""
Test script to verify the implementation of the baseline source notebook
in the .py files. This script tests data loading, model creation, and training.
"""

import torch
import numpy as np
import os
import sys
from tqdm import tqdm

# Import our modules
from loader import load_dti_dataset, process_dti_data, create_data_loaders, custom_collate_fn
from models import CATDTIpy
from loss import DTILoss, calculate_metrics
from args import get_args

def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    try:
        # Test with a small subset
        df = load_dti_dataset(max_samples=100)
        print(f"✅ Data loading successful! Dataset shape: {df.shape}")
        
        # Test data processing
        processed_data = process_dti_data(df, max_samples=50)
        print(f"✅ Data processing successful! Processed {len(processed_data)} samples")
        
        # Test data loaders
        train_data, val_data, test_data = create_data_loaders(processed_data, batch_size=8)
        print(f"✅ Data loaders created! Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Test batching
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, collate_fn=custom_collate_fn)
        batch = next(iter(train_loader))
        print(f"✅ Batching successful! Batch keys: {list(batch.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        # Create model
        model = CATDTIpy()
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test forward pass with dummy data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy batch with proper PyTorch Geometric Data objects
        batch_size = 4
        from torch_geometric.data import Data, Batch
        
        # Create dummy drug graphs
        drug_graphs = []
        for i in range(batch_size):
            # Create a simple graph for each drug
            num_nodes = 50  # Random number of nodes
            x = torch.randn(num_nodes, 75).to(device)  # Node features
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2)).to(device)  # Random edges
            batch_idx = torch.full((num_nodes,), i, dtype=torch.long).to(device)  # Batch indices
            
            graph_data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            drug_graphs.append(graph_data)
        
        # Batch the drug graphs
        batched_drug_graphs = Batch.from_data_list(drug_graphs)
        
        dummy_batch = {
            'drug_graphs': batched_drug_graphs,  # Proper PyTorch Geometric Data object
            'protein_encoded': torch.randint(0, 26, (batch_size, 1000)).to(device),  # Protein sequence
            'protein_mask': torch.ones(batch_size, 1000).to(device),  # Protein mask
            'labels': torch.randint(0, 2, (batch_size,)).float().to(device)  # Labels
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_batch)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test loss functions"""
    print("\nTesting loss functions...")
    
    try:
        # Create dummy predictions and targets
        predictions = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        
        # Test different loss functions
        bce_loss = DTILoss(loss_type='bce')
        focal_loss = DTILoss(loss_type='focal')
        
        bce_value = bce_loss(predictions, targets)
        focal_value = focal_loss(predictions, targets)
        
        print(f"✅ Loss functions work! BCE: {bce_value.item():.4f}, Focal: {focal_value.item():.4f}")
        
        # Test metrics calculation
        metrics = calculate_metrics(predictions, targets)
        print(f"✅ Metrics calculation successful!")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss functions failed: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    try:
        # Load small dataset
        df = load_dti_dataset(max_samples=50)
        processed_data = process_dti_data(df, max_samples=20)
        train_data, _, _ = create_data_loaders(processed_data, batch_size=4)
        
        # Create model and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CATDTIpy().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = DTILoss(loss_type='bce')
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, collate_fn=custom_collate_fn)
        
        # Single training step
        model.train()
        batch = next(iter(train_loader))
        
        # Move batch to device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'].float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful! Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_args():
    """Test argument parsing"""
    print("\nTesting argument parsing...")
    
    try:
        # Test default arguments
        args = get_args()
        print(f"✅ Default arguments loaded successfully!")
        print(f"   Dataset path: {args.dataset_path}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")
        print(f"   Device: {args.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CATDTI Implementation")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Loss Functions", test_loss_functions),
        ("Training Step", test_training_step),
        ("Argument Parsing", test_args)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 