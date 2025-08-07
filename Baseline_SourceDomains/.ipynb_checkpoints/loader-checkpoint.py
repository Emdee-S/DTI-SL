import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom featurizers
try:
    from canonical_featurizers import featurize_molecule
except ImportError as e:
    print(f"Warning: Could not import canonical_featurizers: {e}")

def process_protein_sequence(protein_seq, max_length=1000):
    """
    Process protein sequence similar to dataloader.py
    """
    # Character to integer mapping for amino acids
    CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                   "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                   "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                   "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

    def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
        X = np.zeros(MAX_SEQ_LEN, np.int64)
        for i, ch in enumerate(line[:MAX_SEQ_LEN]):
            X[i] = smi_ch_ind.get(ch, 0)  # Use 0 for unknown characters
        return X

    # Process protein sequence
    pro_len = len(protein_seq)
    protein_encoded = label_sequence(protein_seq, CHARPROTSET, max_length)

    # Create protein mask
    protein_mask = np.zeros(max_length)
    if pro_len > max_length:
        protein_mask[:] = 1
    else:
        protein_mask[:pro_len] = 1

    return protein_encoded, protein_mask, pro_len

def create_molecular_graph(smiles, max_nodes=290):
    """
    Create PyTorch Geometric graph from SMILES using canonical_featurizers
    """
    try:
        # Use canonical featurizers
        atom_features, bond_features = featurize_molecule(smiles, self_loop=True)

        # Extract features
        node_features = torch.tensor(atom_features['h'], dtype=torch.float32)
        edge_features = torch.tensor(bond_features['e'], dtype=torch.float32)
        edge_index = torch.tensor(bond_features['edge_indices'], dtype=torch.long)

        # Add virtual node indicator (similar to dataloader.py)
        num_actual_nodes = node_features.shape[0]
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        node_features = torch.cat((node_features, virtual_node_bit), 1)

        # Add virtual nodes for padding (similar to dataloader.py)
        num_virtual_nodes = max_nodes - num_actual_nodes
        if num_virtual_nodes > 0:
            virtual_node_feat = torch.cat((
                torch.zeros(num_virtual_nodes, 74),  # 74 atom features
                torch.ones(num_virtual_nodes, 1)     # Virtual node indicator
            ), 1)
            node_features = torch.cat((node_features, virtual_node_feat), 0)

            # Add self-loops for virtual nodes
            virtual_self_loops = torch.stack([
                torch.arange(num_actual_nodes, max_nodes),
                torch.arange(num_actual_nodes, max_nodes)
            ], dim=0)
            edge_index = torch.cat((edge_index, virtual_self_loops), 1)

        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=max_nodes
        )

        return graph_data, num_actual_nodes

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        # Return empty graph as fallback
        empty_features = torch.zeros(max_nodes, 75)  # 74 + 1 virtual indicator
        empty_features[:, -1] = 1  # Mark all as virtual
        empty_edges = torch.stack([torch.arange(max_nodes), torch.arange(max_nodes)], dim=0)
        empty_edge_features = torch.zeros(max_nodes, 13)  # 12 + 1 for self-loops

        return Data(
            x=empty_features,
            edge_index=empty_edges,
            edge_attr=empty_edge_features,
            num_nodes=max_nodes
        ), 0

def load_dti_dataset(dataset_path=None, max_samples=None):
    """Load DTI dataset from specified path or use default"""
    
    print("Loading DTI datasets...")
    
    try:
        if dataset_path is None:
            # Default dataset paths
            base_dir = 'datasets/human'  # Options: human, biosnap, bindingdb
            dataset_path = base_dir
            
        train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
        val_df = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
        test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

        print(f"✅ Original dataset shapes: Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

        # Concatenate all datasets in the specified order: test, train, val
        combined_df = pd.concat([test_df, train_df, val_df], ignore_index=True)
        print(f"✅ Concatenated Test, Train, and Val shape: {combined_df.shape}")

        # Take a manageable chunk from the combined set if max_samples is specified
        if max_samples and len(combined_df) > max_samples:
            combined_df = combined_df.head(max_samples)
            print(f"Using subset of {max_samples} samples for faster processing")

        print(f"✅ Final combined dataset shape: {combined_df.shape}")
        print(f"Class distribution:\n{combined_df['Y'].value_counts()}")
        print(f"Class balance: {combined_df['Y'].value_counts(normalize=True).round(3)}")

        # Show some statistics for the combined set
        print(f"\n Combined Dataset Statistics:")
        print(f"Total drug-protein pairs: {len(combined_df)}")
        print(f"Average SMILES length: {combined_df['SMILES'].str.len().mean():.1f} characters")
        print(f"Average protein length: {combined_df['Protein'].str.len().mean():.1f} characters")
        print(f"Shortest SMILES: {combined_df['SMILES'].str.len().min()} characters")
        print(f"Longest SMILES: {combined_df['SMILES'].str.len().max()} characters")
        print(f"Shortest protein: {combined_df['Protein'].str.len().min()} characters")
        print(f"Longest protein: {combined_df['Protein'].str.len().max()} characters")

        return combined_df

    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        print("Creating fallback synthetic dataset...")

        # Fallback synthetic dataset
        dti_data = {
            'SMILES': [
                'CCO', 'CC(=O)O', 'c1ccccc1', 'CCN(CC)CC', 'CC(C)O',
                'c1ccc2ccccc2c1', 'CCCCO', 'CC(C)(C)O', 'CC(C)C', 'CCCC'
            ],
            'Protein': [
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKLIVLCSVAVILMGTFMLTFLTQKKAKQRGLL',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKLIVLCSVAVILMGTFMLTFLTQKKAKQRGLL',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKLIVLCSVAVILMGTFMLTFLTQKKAKQRGLL',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKLIVLCSVAVILMGTFMLTFLTQKKAKQRGLL',
                'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
                'MKLIVLCSVAVILMGTFMLTFLTQKKAKQRGLL'
            ],
            'Y': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(dti_data)
        print(f"✅ Fallback dataset created. Shape: {df.shape}")
        return df

def process_dti_data(df, max_samples=None, max_nodes=290, max_protein_length=1000):
    """
    Process DTI dataset similar to dataloader.py but for PyTorch Geometric
    """
    print("Processing DTI data...")

    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"Using subset of {max_samples} samples")

    processed_data = []

    for idx, row in df.iterrows():
        try:
            # Process drug (SMILES) as graph
            smiles = row['SMILES']
            drug_graph, num_actual_nodes = create_molecular_graph(smiles, max_nodes)

            # Process protein sequence
            protein_seq = row['Protein']
            protein_encoded, protein_mask, pro_len = process_protein_sequence(
                protein_seq, max_protein_length
            )

            # Convert to tensors
            protein_encoded = torch.tensor(protein_encoded, dtype=torch.long)
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            label = torch.tensor(row['Y'], dtype=torch.float32)

            # Store processed data
            sample = {
                'drug_graph': drug_graph,
                'protein_encoded': protein_encoded,
                'protein_mask': protein_mask,
                'protein_length': pro_len,
                'label': label,
                'smiles': smiles,
                'protein_seq': protein_seq,
                'num_actual_nodes': num_actual_nodes
            }

            processed_data.append(sample)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"✅ Successfully processed {len(processed_data)} samples")
    return processed_data

def create_data_loaders(processed_data, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/val/test data loaders
    """
    print("Creating data loaders...")

    # Split data
    total_samples = len(processed_data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size + val_size]
    test_data = processed_data[train_size + val_size:]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data

def custom_collate_fn(batch):
    """
    Custom collate function for batching graphs and sequences
    """
    # Separate different data types
    drug_graphs = [item['drug_graph'] for item in batch]
    protein_encoded = torch.stack([item['protein_encoded'] for item in batch])
    protein_mask = torch.stack([item['protein_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Batch the graphs
    batched_graphs = Batch.from_data_list(drug_graphs)

    return {
        'drug_graphs': batched_graphs,
        'protein_encoded': protein_encoded,
        'protein_mask': protein_mask,
        'labels': labels
    }

def move_batch_to_device(batch, device):
    """Move batch data to specified device"""
    new_batch = {}
    for k, v in batch.items():
        if hasattr(v, 'to'):
            new_batch[k] = v.to(device)
        elif isinstance(v, dict):
            new_batch[k] = move_batch_to_device(v, device)
        else:
            new_batch[k] = v
    return new_batch
