import argparse
import os
import torch

def get_args():
    """Get command line arguments for DTI training"""
    
    parser = argparse.ArgumentParser(description='CATDTI: Drug-Target Interaction Prediction')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, default='datasets/human',
                       help='Path to dataset directory (human, biosnap, bindingdb)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for faster testing)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of validation data')
    
    # Model arguments
    parser.add_argument('--drug_node_feat_dim', type=int, default=75,
                       help='Drug node feature dimension')
    parser.add_argument('--drug_embedding', type=int, default=128,
                       help='Drug embedding dimension')
    parser.add_argument('--drug_hidden_feats', nargs='+', type=int, default=[128, 128, 128],
                       help='Drug GNN hidden feature dimensions')
    parser.add_argument('--protein_vocab_size', type=int, default=26,
                       help='Protein vocabulary size')
    parser.add_argument('--protein_emb_dim', type=int, default=128,
                       help='Protein embedding dimension')
    parser.add_argument('--protein_max_len', type=int, default=1000,
                       help='Maximum protein sequence length')
    parser.add_argument('--protein_num_layers', type=int, default=3,
                       help='Number of protein encoder layers')
    parser.add_argument('--protein_num_attention_heads', type=int, default=8,
                       help='Number of attention heads in protein encoder')
    parser.add_argument('--protein_ff_expansion', type=int, default=4,
                       help='Protein feed-forward expansion factor')
    parser.add_argument('--protein_ff_dropout', type=float, default=0.1,
                       help='Protein feed-forward dropout rate')
    parser.add_argument('--protein_attn_dropout', type=float, default=0.1,
                       help='Protein attention dropout rate')
    parser.add_argument('--protein_conv_dropout', type=float, default=0.1,
                       help='Protein convolution dropout rate')
    parser.add_argument('--protein_conv_kernel', type=int, default=3,
                       help='Protein convolution kernel size')
    parser.add_argument('--mlp_in_dim', type=int, default=256,
                       help='MLP input dimension')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                       help='MLP hidden dimension')
    parser.add_argument('--mlp_out_dim', type=int, default=128,
                       help='MLP output dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='none',
                       choices=['none', 'step', 'cosine', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--scheduler_step_size', type=int, default=30,
                       help='Step size for step scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                       help='Gamma for step scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                       help='Patience for plateau scheduler')
    
    # Loss arguments
    parser.add_argument('--loss_type', type=str, default='bce',
                       choices=['bce', 'focal', 'combined'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--bce_weight', type=float, default=1.0,
                       help='Weight for BCE loss in combined loss')
    parser.add_argument('--focal_weight', type=float, default=0.0,
                       help='Weight for focal loss in combined loss')
    parser.add_argument('--consistency_weight', type=float, default=0.0,
                       help='Weight for consistency loss in combined loss')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--nesterov', action='store_true',
                       help='Use Nesterov momentum for SGD')
    
    # Regularization arguments
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    
    # Evaluation arguments
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Evaluation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save interval (epochs)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-4,
                       help='Minimum delta for early stopping')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints and results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, or specific GPU)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                       help='Pin memory for faster data loading')
    
    # Logging arguments
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (steps)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Use TensorBoard for logging')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='catdti',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Weights & Biases entity name')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (smaller dataset, fewer epochs)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set default experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"catdti_{args.dataset_path.split('/')[-1]}_{args.loss_type}"
    
    # Create save directory
    args.save_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Debug mode adjustments
    if args.debug:
        args.max_samples = 1000
        args.epochs = 5
        args.batch_size = 16
        args.eval_interval = 1
        args.save_interval = 2
    
    return args

def save_args(args, save_path):
    """Save arguments to a JSON file"""
    import json
    
    # Convert args to dictionary
    args_dict = vars(args)
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(args_dict, f, indent=2)

def load_args(load_path):
    """Load arguments from a JSON file"""
    import json
    
    with open(load_path, 'r') as f:
        args_dict = json.load(f)
    
    # Create a new parser and set the arguments
    parser = argparse.ArgumentParser()
    for key, value in args_dict.items():
        parser.add_argument(f'--{key}', default=value)
    
    return parser.parse_args([])

if __name__ == "__main__":
    # Test argument parsing
    args = get_args()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Test saving and loading
    save_args(args, 'test_args.json')
    loaded_args = load_args('test_args.json')
    print("\nLoaded arguments:")
    for key, value in vars(loaded_args).items():
        print(f"  {key}: {value}")
