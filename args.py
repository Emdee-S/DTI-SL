# arguments for general hyperparameters, proteins, and smiles
class Args:
    n_epochs   = 10
    batch_size = 32
    lr         = 0.0001
    n_cpu      = 2
    shuffle    = True
    reg        = 0.0001
    drop       = 0.1
    S          = 'bindingdb'
    T          = 'biosnap'
    
args = Args()

# there are many more arguments to be added for both smiles and proteins later
class Prot_Args: 
    max = 1000
    encode_dim = 512
    layers = 3
    attention_heads = 8
    
prot_args = Prot_Args()

class Smiles_Args:
    max_nodes = 290
    
smiles_args = Smiles_Args()