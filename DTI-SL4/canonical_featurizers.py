# -*- coding: utf-8 -*-
# Canonical Atom and Bond Featurizers - extracted from DGL-LifeSci
# Dependencies: rdkit, numpy

import itertools
from collections import defaultdict
import numpy as np

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit is required. Install with: pip install rdkit")

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.
    
    Parameters
    ----------
    x
        Value to encode.
    allowable_set : list
        The elements of the allowable_set should be of the same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the additional last element.
    
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)
    
    if encode_unknown and (x not in allowable_set):
        x = None
    
    return list(map(lambda s: x == s, allowable_set))

#################################################################
# Atom featurization functions
#################################################################

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom."""
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom."""
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom."""
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get the formal charge of an atom."""
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    """Get the number of radical electrons of an atom."""
    return [atom.GetNumRadicalElectrons()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom."""
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic."""
    return [atom.GetIsAromatic()]

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs on the atom."""
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

#################################################################
# Bond featurization functions
#################################################################

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond."""
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """Get whether the bond is conjugated."""
    return [bond.GetIsConjugated()]

def bond_is_in_ring(bond):
    """Get whether the bond is in a ring of any size."""
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the stereo configuration of a bond."""
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

#################################################################
# Featurizer classes
#################################################################

class ConcatFeaturizer(object):
    """Concatenate the evaluation results of multiple functions as a single feature."""
    
    def __init__(self, func_list):
        self.func_list = func_list
    
    def __call__(self, x):
        """Featurize the input data."""
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class CanonicalAtomFeaturizer(object):
    """Canonical atom featurizer with standard features.
    
    The atom features include:
    * One hot encoding of the atom type (43 atom types)
    * One hot encoding of the atom degree (0-10)
    * One hot encoding of the number of implicit Hs on the atom (0-6)
    * Formal charge of the atom
    * Number of radical electrons of the atom
    * One hot encoding of the atom hybridization (SP, SP2, SP3, SP3D, SP3D2)
    * Whether the atom is aromatic
    * One hot encoding of the number of total Hs on the atom (0-4)
    
    Total: 74 features
    """
    
    def __init__(self, atom_data_field='h'):
        self.atom_data_field = atom_data_field
        self.featurizer_funcs = {
            atom_data_field: ConcatFeaturizer([
                atom_type_one_hot,
                atom_degree_one_hot,
                atom_implicit_valence_one_hot,
                atom_formal_charge,
                atom_num_radical_electrons,
                atom_hybridization_one_hot,
                atom_is_aromatic,
                atom_total_num_H_one_hot
            ])
        }
    
    def feat_size(self, feat_name=None):
        """Get the feature size for feat_name."""
        if feat_name is None:
            feat_name = self.atom_data_field
        
        if feat_name not in self.featurizer_funcs:
            raise ValueError(f'Feature {feat_name} not found')
        
        # Calculate feature size by testing on a simple molecule
        atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
        return len(self.featurizer_funcs[feat_name](atom))
    
    def __call__(self, mol):
        """Featurize all atoms in a molecule.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        
        Returns
        -------
        dict
            Dictionary with atom features as numpy array of shape (num_atoms, 74)
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)
        
        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))
        
        # Stack the features and convert them to numpy arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = feat.astype(np.float32)
        
        return processed_features

class CanonicalBondFeaturizer(object):
    """Canonical bond featurizer with standard features.
    
    The bond features include:
    * One hot encoding of the bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    * Whether the bond is conjugated
    * Whether the bond is in a ring of any size
    * One hot encoding of the stereo configuration of a bond (6 stereo types)
    
    Total: 12 features
    """
    
    def __init__(self, bond_data_field='e', self_loop=False):
        self.bond_data_field = bond_data_field
        self.self_loop = self_loop
        self.featurizer_funcs = {
            bond_data_field: ConcatFeaturizer([
                bond_type_one_hot,
                bond_is_conjugated,
                bond_is_in_ring,
                bond_stereo_one_hot
            ])
        }
    
    def feat_size(self, feat_name=None):
        """Get the feature size for feat_name."""
        if feat_name is None:
            feat_name = self.bond_data_field
        
        if feat_name not in self.featurizer_funcs:
            raise ValueError(f'Feature {feat_name} not found')
        
        # Calculate feature size by testing on a simple molecule
        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)
        return feats[feat_name].shape[1]
    
    def __call__(self, mol):
        """Featurize all bonds in a molecule.
        
        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.
        
        Returns
        -------
        dict
            Dictionary with bond features as numpy array of shape (num_bonds*2, 12)
            and edge_indices as numpy array of shape (2, num_bonds*2)
            Note: Each bond appears twice (for bidirectional graph)
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)
        edge_indices = []
        
        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])  # Duplicate for bidirectional
            
            # Add bidirectional edges
            edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        
        # Stack the features and convert them to numpy arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = feat.astype(np.float32)
        
        # Convert edge indices to numpy array
        if edge_indices:
            processed_features['edge_indices'] = np.array(edge_indices, dtype=np.int64).T
        else:
            processed_features['edge_indices'] = np.array([[], []], dtype=np.int64)
        
        # Handle self loops if requested
        if self.self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                if feat_name == 'edge_indices':
                    continue
                feats = processed_features[feat_name]
                feats = np.concatenate([feats, np.zeros((feats.shape[0], 1))], axis=1)
                self_loop_feats = np.zeros((num_atoms, feats.shape[1]))
                self_loop_feats[:, -1] = 1
                feats = np.concatenate([feats, self_loop_feats], axis=0)
                processed_features[feat_name] = feats
            
            # Add self-loop edges
            if 'edge_indices' in processed_features:
                self_loop_edges = np.array([[i, i] for i in range(num_atoms)], dtype=np.int64).T
                if processed_features['edge_indices'].size > 0:
                    processed_features['edge_indices'] = np.concatenate(
                        [processed_features['edge_indices'], self_loop_edges], axis=1
                    )
                else:
                    processed_features['edge_indices'] = self_loop_edges
        
        if self.self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                if feat_name == 'edge_indices':
                    continue
                feats = processed_features[feat_name]
                feats = np.zeros((num_atoms, feats.shape[1]))
                feats[:, -1] = 1
                processed_features[feat_name] = feats
            
            # Add self-loop edges for molecules with no bonds
            if num_atoms > 0:
                self_loop_edges = np.array([[i, i] for i in range(num_atoms)], dtype=np.int64).T
                processed_features['edge_indices'] = self_loop_edges
        
        return processed_features

#################################################################
# Utility function for easy usage
#################################################################

def featurize_molecule(smiles, atom_data_field='h', bond_data_field='e', self_loop=False):
    """
    Featurize a molecule from SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the molecule
    atom_data_field : str
        Name for atom features in output dictionary
    bond_data_field : str
        Name for bond features in output dictionary
    self_loop : bool
        Whether to add self loops to bonds
    
    Returns
    -------
    tuple
        (atom_features, bond_features) as dictionaries with numpy arrays
        bond_features now includes 'edge_indices' key with shape (2, num_edges)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field)
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field, self_loop)
    
    atom_features = atom_featurizer(mol)
    bond_features = bond_featurizer(mol)
    
    return atom_features, bond_features

# Example usage
if __name__ == "__main__":
    # Test the featurizers
    smiles = "CCO"  # Ethanol
    
    print(f"Featurizing: {smiles}")
    
    # Get features
    atom_features, bond_features = featurize_molecule(smiles)
    
    print(f"Atom features shape: {atom_features['h'].shape}")
    print(f"Bond features shape: {bond_features['e'].shape}")
    print(f"Edge indices shape: {bond_features['edge_indices'].shape}")
    print(f"Atom feature size: {len(atom_features['h'][0])}")
    print(f"Bond feature size: {len(bond_features['e'][0])}")
    
    print(f"\nFirst atom features (first 10): {atom_features['h'][0][:10]}")
    print(f"First bond features: {bond_features['e'][0]}")
    print(f"Edge indices: {bond_features['edge_indices']}")
    
    # Test feature sizes
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer()
    
    print(f"\nAtom featurizer feature size: {atom_featurizer.feat_size()}")
    print(f"Bond featurizer feature size: {bond_featurizer.feat_size()}")
    
    # Test with self loops
    print(f"\nTesting with self loops...")
    atom_features_self, bond_features_self = featurize_molecule(smiles, self_loop=True)
    print(f"With self loops - Edge indices shape: {bond_features_self['edge_indices'].shape}")
    print(f"With self loops - Edge indices: {bond_features_self['edge_indices']}")
    
    # Test with a single atom molecule
    print(f"\nTesting single atom molecule...")
    atom_features_single, bond_features_single = featurize_molecule("C", self_loop=True)
    print(f"Single atom - Edge indices shape: {bond_features_single['edge_indices'].shape}")
    print(f"Single atom - Edge indices: {bond_features_single['edge_indices']}") 