import pandas as pd
import torch.utils.data as data
import torch
import numpy as np 
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein


# ---------- amino-acid dictionary ----------
amino_dict = {"B": 1, "A": 2, "C": 3, "E": 4, "D": 5, "G": 6,
              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
              "U": 19, "T": 20, "W": 21, "V": 22, "X": 23, "Z": 24, "Y": 25}

# ---------- helper: encode one protein ----------
def encode_protein(seq: str, max_len: int):
    encoded = [amino_dict.get(res, 0) for res in seq[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([0] * (max_len - len(encoded)))
    return np.asarray(encoded, dtype=np.int16)




# TODO data loading class for drug,protein pairs, essentially want one block to call all thats needed to encode proteins,smiles and output them in the way we want
class DrugProteinDataset(Dataset):
    """
    A lightweight Dataset that:
      • stores pre-encoded proteins (np.ndarray of shape [N, max_len])
      • builds/pads drug graphs on demand
      • returns tensors ready for model input
    """
    def __init__(self, df: pd.DataFrame,
                 prot_args: Prot_Args,
                 smiles_args: Smiles_Args):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.p_max = prot_args.max # max is 1000 amino acids
        self.d_max = smiles_args.max_nodes # max is 290 nodes in the graph

        # — pre-encode every protein once —
        self.protein_int = np.stack(
            df["Protein"].apply(lambda s: encode_protein(s, self.p_max)).values
        )

        # — drug graph helpers —
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.bigraph_fn = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- DRUG / SMILES --------
        g = self.bigraph_fn(
            smiles=row["SMILES"],
            node_featurizer=self.atom_featurizer,
            edge_featurizer=self.bond_featurizer,
        )

        # real-node indicator before padding
        n_real = g.num_nodes()
        real_node_bit = torch.zeros(n_real, 1)
        g.ndata["h"] = torch.cat([g.ndata.pop("h"), real_node_bit], dim=1) # h gets node features, e for bond features

        # pad to self.d_max virtual nodes
        n_fake = self.d_max - n_real
        if n_fake < 0:
            raise ValueError(f"SMILES string at index {idx} has {n_real} atoms "
                             f"which exceeds max_nodes={self.d_max}. "
                             "Either increase Smiles_Args.max_nodes or drop this entry.")
        if n_fake:
            virtual_feats = torch.cat([torch.zeros(n_fake, 74),
                                       torch.ones(n_fake, 1)], dim=1)
            g.add_nodes(n_fake, {"h": virtual_feats})

        # -------- PROTEIN --------
        prot_int = torch.tensor(self.protein_int[idx], dtype=torch.long)     # [max_len]
        mask = (prot_int != 0).float()                                       # 1 for real, 0 for pad

        # -------- LABEL --------
        y = torch.tensor(row["Y"], dtype=torch.float32)

        return g, prot_int, mask, y


# TODO maybe data loading class for both domains