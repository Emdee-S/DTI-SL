import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

# --- Absolute Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

# --- Relative MultiHead Attention (from models.py, simplified for batch_first) ---
class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = d_model ** 0.5

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, pos_embedding, mask=None):
        # query, key, value: (batch, seq_len, d_model)
        batch_size, seq_len, _ = query.size()
        # Project and reshape for multi-head
        q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.d_head)
        k = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.d_head)
        p = self.pos_proj(pos_embedding).view(batch_size, seq_len, self.num_heads, self.d_head)

        # Transpose for attention: (batch, num_heads, seq_len, d_head)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        p = p.permute(0, 2, 1, 3)

        # Content-based attention
        content_score = torch.matmul((q + self.u_bias.unsqueeze(0).unsqueeze(2)), k.transpose(-2, -1))
        # Position-based attention
        pos_score = torch.matmul((q + self.v_bias.unsqueeze(0).unsqueeze(2)), p.transpose(-2, -1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            score = score.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(score, -1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        # pos_score: (batch, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len1, seq_len2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_len1, 1)
        padded = torch.cat([zeros, pos_score], dim=-1)
        padded = padded.view(batch_size, num_heads, seq_len2 + 1, seq_len1)
        return padded[:, :, 1:].view_as(pos_score)

# --- FeedForward Module (for Transformer) ---
class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim, expansion_factor=4, dropout_p=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim),
            nn.Dropout(dropout_p)
        )
    def forward(self, x):
        return self.seq(x)

# --- CNN+Transformer Block for Protein ---
class CNNTransBlock(nn.Module):
    def __init__(self, encoder_dim, num_attention_heads, feed_forward_expansion_factor, 
                 feed_forward_dropout_p, attention_dropout_p, conv_dropout_p, conv_kernel_size, max_len):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(encoder_dim)
        self.rel_attn = RelativeMultiHeadAttention(encoder_dim, num_attention_heads)
        self.layernorm2 = nn.LayerNorm(encoder_dim)
        self.conv = nn.Conv1d(encoder_dim, encoder_dim, conv_kernel_size, padding=conv_kernel_size//2)
        self.dropout = nn.Dropout(conv_dropout_p)
        self.ff = FeedForwardModule(encoder_dim, feed_forward_expansion_factor, feed_forward_dropout_p)
        self.layernorm3 = nn.LayerNorm(encoder_dim)
        self.pos_encoder = PositionalEncoding(encoder_dim, max_len)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, encoder_dim)
        pos_emb = self.pos_encoder.pe[:x.size(1), :].unsqueeze(0).expand(x.size(0), -1, -1)
        attn_out = self.rel_attn(x, x, x, pos_emb, mask)
        x = self.layernorm1(x + attn_out)
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv(x_conv)
        x_conv = F.relu(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.permute(0, 2, 1)
        x = self.layernorm2(x + x_conv)
        ff_out = self.ff(x)
        x = self.layernorm3(x + ff_out)
        return x

# --- Protein Encoder ---
class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size=26, emb_dim=128, max_len=1000, num_layers=3, num_attention_heads=8, 
                 feed_forward_expansion_factor=4, feed_forward_dropout_p=0.1, 
                 attention_dropout_p=0.1, conv_dropout_p=0.1, conv_kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.blocks = nn.ModuleList([
            CNNTransBlock(
                encoder_dim=emb_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                max_len=max_len
            ) for _ in range(num_layers)
        ])
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seq, mask=None):
        # seq: (batch, seq_len)
        x = self.embedding(seq)  # (batch, seq_len, emb_dim)
        for block in self.blocks:
            x = block(x, mask)
        # Pool over sequence
        x = x.permute(0, 2, 1)  # (batch, emb_dim, seq_len)
        x = self.maxpool(x).squeeze(-1)  # (batch, emb_dim)
        return x

# --- Molecular GCN for Drug Graphs ---
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, hidden_feats=[128,128,128]):
        super().__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.gnn_layers = nn.ModuleList()
        last_dim = dim_embedding
        for h in hidden_feats:
            self.gnn_layers.append(GCNConv(last_dim, h))
            last_dim = h
        self.output_feats = last_dim

    def forward(self, x, edge_index, batch):
        x = self.init_transform(x)
        for gnn in self.gnn_layers:
            x = F.relu(gnn(x, edge_index))
        # Pool to get (batch, output_feats)
        x = global_max_pool(x, batch)
        return x

# --- MLP Decoder (as in CATDTI) ---
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

# --- Main CATDTI-py Model ---
class CATDTIpy(nn.Module):
    def __init__(self,
                 drug_node_feat_dim=75,
                 drug_embedding=128,
                 drug_hidden_feats=[128,128,128],
                 protein_vocab_size=26,
                 protein_emb_dim=128,
                 protein_max_len=1000,
                 protein_num_layers=3,
                 protein_num_attention_heads=8,
                 protein_ff_expansion=4,
                 protein_ff_dropout=0.1,
                 protein_attn_dropout=0.1,
                 protein_conv_dropout=0.1,
                 protein_conv_kernel=3,
                 mlp_in_dim=256,
                 mlp_hidden_dim=512,
                 mlp_out_dim=128,
                 out_binary=1):
        super().__init__()
        # Drug GNN
        self.drug_extractor = MolecularGCN(
            in_feats=drug_node_feat_dim,
            dim_embedding=drug_embedding,
            hidden_feats=drug_hidden_feats
        )
        # Protein encoder
        self.protein_encoder = ProteinEncoder(
            vocab_size=protein_vocab_size,
            emb_dim=protein_emb_dim,
            max_len=protein_max_len,
            num_layers=protein_num_layers,
            num_attention_heads=protein_num_attention_heads,
            feed_forward_expansion_factor=protein_ff_expansion,
            feed_forward_dropout_p=protein_ff_dropout,
            attention_dropout_p=protein_attn_dropout,
            conv_dropout_p=protein_conv_dropout,
            conv_kernel_size=protein_conv_kernel
        )
        # Multihead Attention (absolute, for cross-modality)
        self.mix_attention_layer = nn.MultiheadAttention(protein_emb_dim, 4, batch_first=True)
        # Dropout and MLP
        self.dropout1 = nn.Dropout(0.1)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, batch, mode="train"):
        # Drug graph
        x, edge_index, batch_idx = batch['drug_graphs'].x, batch['drug_graphs'].edge_index, batch['drug_graphs'].batch
        v_d = self.drug_extractor(x, edge_index, batch_idx)  # (batch, gnn_dim)
        v_d_exp = v_d.unsqueeze(1)  # (batch, 1, gnn_dim) for attention

        # Protein
        v_p = self.protein_encoder(batch['protein_encoded'].long(), batch['protein_mask'].long())  # (batch, emb_dim)
        v_p_exp = v_p.unsqueeze(1)  # (batch, 1, emb_dim) for attention

        # Multihead Attention (drug as query, protein as key/value)
        drug_att, _ = self.mix_attention_layer(v_d_exp, v_p_exp, v_p_exp)
        drug_att = drug_att.squeeze(1)  # (batch, emb_dim)
        # Combine original and attended features
        drug_final = 0.5 * v_d + 0.5 * drug_att
        protein_final = v_p  # (batch, emb_dim)
        # Concatenate and classify
        pair = torch.cat([drug_final, protein_final], dim=1)
        pair = self.dropout1(pair)
        score = self.mlp_classifier(pair)
        return score.squeeze(-1)