# CATDTI: Drug-Target Interaction Prediction

This repository contains a PyTorch implementation of the CATDTI (Convolutional Attention Transformer for Drug-Target Interaction) model, converted from the baseline Jupyter notebook into modular Python files.

## 📁 Project Structure

```
DTI-SL5/
├── args.py                 # Command-line arguments and configuration
├── canonical_featurizers.py # Molecular featurization (already implemented)
├── loader.py              # Data loading and processing
├── models.py              # Neural network models
├── loss.py                # Loss functions and metrics
├── main.py                # Main training script
├── test_implementation.py # Test script to verify implementation
├── README.md              # This file
└── datasets/              # Dataset directory
    ├── human/
    ├── biosnap/
    └── bindingdb/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch==2.7.1 torch-geometric==2.6.1 scikit-learn matplotlib tqdm pandas requests rdkit-pypi
```

### 2. Test the Implementation

```bash
python test_implementation.py
```

### 3. Train the Model

```bash
# Basic training with default parameters
python main.py

# Training with custom parameters
python main.py --dataset_path datasets/human --batch_size 64 --epochs 100 --lr 1e-4

# Debug mode (smaller dataset, fewer epochs)
python main.py --debug

# Resume from checkpoint
python main.py --resume checkpoints/best_model.pth
```

## 📊 Model Architecture

The CATDTI model consists of three main components:

### 1. Drug Encoder (Molecular GCN)
- **Input**: SMILES strings converted to molecular graphs
- **Architecture**: Graph Convolutional Network (GCN) with 3 layers
- **Output**: 128-dimensional drug embeddings

### 2. Protein Encoder (CNN + Transformer)
- **Input**: Protein sequences encoded as amino acid tokens
- **Architecture**: 
  - Embedding layer (26 vocab size → 128 dim)
  - 3 CNN+Transformer blocks with relative attention
  - Global max pooling
- **Output**: 128-dimensional protein embeddings

### 3. Interaction Predictor
- **Input**: Concatenated drug and protein embeddings
- **Architecture**: Multi-layer perceptron with batch normalization
- **Output**: Binary interaction probability

## 🔧 Configuration

### Command Line Arguments

#### Data Arguments
- `--dataset_path`: Path to dataset directory (human, biosnap, bindingdb)
- `--max_samples`: Maximum number of samples to use
- `--train_ratio`: Ratio of training data (default: 0.8)
- `--val_ratio`: Ratio of validation data (default: 0.1)

#### Model Arguments
- `--drug_node_feat_dim`: Drug node feature dimension (default: 75)
- `--drug_embedding`: Drug embedding dimension (default: 128)
- `--protein_emb_dim`: Protein embedding dimension (default: 128)
- `--protein_max_len`: Maximum protein sequence length (default: 1000)
- `--protein_num_layers`: Number of protein encoder layers (default: 3)

#### Training Arguments
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 5e-5)
- `--optimizer`: Optimizer type (adam, sgd, adamw)
- `--scheduler`: Learning rate scheduler (none, step, cosine, plateau)

#### Loss Arguments
- `--loss_type`: Loss function type (bce, focal, combined)
- `--focal_alpha`: Alpha parameter for focal loss
- `--focal_gamma`: Gamma parameter for focal loss

### Example Configurations

```bash
# High-performance training
python main.py \
    --dataset_path datasets/human \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-4 \
    --scheduler cosine \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0

# Quick testing
python main.py \
    --debug \
    --max_samples 1000 \
    --epochs 10 \
    --batch_size 16
```

## 📈 Training and Evaluation

### Training Process

1. **Data Loading**: Loads DTI datasets from CSV files
2. **Data Processing**: 
   - Converts SMILES to molecular graphs using RDKit
   - Encodes protein sequences using amino acid vocabulary
   - Creates PyTorch Geometric data objects
3. **Model Training**:
   - Trains with specified loss function and optimizer
   - Validates on validation set
   - Saves best model based on validation AUC
   - Supports early stopping and learning rate scheduling

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **AUC**: Area Under the ROC Curve
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Output Files

Training creates the following files in the checkpoint directory:

- `best_model.pth`: Best model based on validation AUC
- `checkpoint_epoch_X.pth`: Checkpoints at specified intervals
- `args.json`: Training configuration
- `results.json`: Final test results and training history
- `training_history.png`: Training curves plot

## 🧪 Testing

Run the test script to verify all components work correctly:

```bash
python test_implementation.py
```

This will test:
- ✅ Data loading and processing
- ✅ Model creation and forward pass
- ✅ Loss functions and metrics
- ✅ Training step
- ✅ Argument parsing

## 🔍 Key Features

### 1. Modular Design
- **loader.py**: Handles all data loading and preprocessing
- **models.py**: Contains all neural network architectures
- **loss.py**: Implements various loss functions and metrics
- **main.py**: Orchestrates training and evaluation

### 2. Flexible Configuration
- Comprehensive command-line argument system
- Support for different datasets (human, biosnap, bindingdb)
- Configurable model architectures
- Multiple loss functions and optimizers

### 3. Robust Training
- Early stopping to prevent overfitting
- Learning rate scheduling
- Checkpoint saving and resuming
- Comprehensive logging and visualization

### 4. Production Ready
- Error handling and fallback mechanisms
- Reproducible results with seed setting
- GPU/CPU compatibility
- Memory-efficient data loading

## 📋 Requirements

- Python 3.7+
- PyTorch 2.7.1
- PyTorch Geometric 2.6.1
- RDKit
- scikit-learn
- matplotlib
- tqdm
- pandas

## 🎯 Performance

Based on the notebook implementation, the model achieves:
- **Test Accuracy**: ~91.8%
- **Test AUC**: ~96.1%
- **Test F1**: ~90.1%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Original CATDTI paper and implementation
- PyTorch Geometric for graph neural networks
- RDKit for molecular processing
- The baseline notebook that served as the foundation for this implementation 