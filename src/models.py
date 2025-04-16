import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import math # For positional encoding

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if max_len <= 0:
             raise ValueError(f"max_len must be positive, got {max_len}")
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        if d_model % 2 != 0:
            num_pairs = d_model // 2
            div_term = torch.exp(torch.arange(0, num_pairs * 2, 2) * (-math.log(10000.0) / (num_pairs * 2)))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0:num_pairs*2:2] = torch.sin(position * div_term)
            pe[:, 1:num_pairs*2:2] = torch.cos(position * div_term)
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Args: x: Tensor, shape [seq_len, batch_size, embedding_dim] """
        if x.size(0) > self.pe.size(0):
            raise ValueError(f"Input sequence length ({x.size(0)}) exceeds PositionalEncoding max_len ({self.pe.size(0)})")
        pos_encoding_slice = self.pe[:x.size(0), :]
        x = x + pos_encoding_slice
        return self.dropout(x)

# --- Transformer Encoder for Time Series ---
class TimeSeriesTransformerEncoder(nn.Module):
    def __init__(self, config, num_regions, max_len=5000): # Use num_regions for input_feat_dim
        super(TimeSeriesTransformerEncoder, self).__init__()
        model_cfg = config['model_params']
        self.d_model = model_cfg['ts_transformer_d_model']
        nhead = model_cfg['ts_transformer_nhead']
        num_encoder_layers = model_cfg['ts_transformer_num_layers']
        dim_feedforward = model_cfg['ts_transformer_dim_feedforward']
        dropout = model_cfg['dropout_rate']
        output_dim = model_cfg['ts_embedding_dim']

        if self.d_model % nhead != 0:
             raise ValueError(f"Transformer d_model ({self.d_model}) must be divisible by nhead ({nhead})")
        if num_regions <= 0:
            raise ValueError(f"num_regions must be positive, got {num_regions}")

        # Input projection now takes num_regions as input dimension
        self.input_projection = nn.Sequential(
            nn.Linear(num_regions, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model)
        )

        self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=False, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(self.d_model, output_dim)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for name, p in self.named_parameters():
            if 'bias' in name: nn.init.zeros_(p)

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: shape [seq_len, batch_size, num_regions]
            src_key_padding_mask: shape [batch_size, seq_len]. BoolTensor (True = padded).
        Returns:
            shape [batch_size, output_dim]
        """
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        src = self.input_projection(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        if src_key_padding_mask is not None:
            valid_mask_batch_first = ~src_key_padding_mask
            valid_mask = valid_mask_batch_first.transpose(0, 1).unsqueeze(-1).float()
            memory = memory * valid_mask
            summed_output = memory.sum(dim=0)
            valid_count = valid_mask.sum(dim=0).clamp(min=1.0)
            pooled_output = summed_output / valid_count
        else:
             pooled_output = memory.mean(dim=0)

        output_embedding = self.output_layer(pooled_output)
        return output_embedding


# --- PhenotypeEncoder ---
class PhenotypeEncoder(nn.Module):
    """Simple MLP for phenotype features."""
    def __init__(self, config, input_dim):
        super(PhenotypeEncoder, self).__init__()
        model_cfg = config['model_params']
        output_dim = model_cfg['pheno_embedding_dim']
        hidden_dim = model_cfg['pheno_hidden_dim']
        dropout = model_cfg['dropout_rate']

        if input_dim <= 0:
             self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x):
        if x.nelement() == 0 and isinstance(self.net, nn.Identity):
             output_dim = config['model_params']['pheno_embedding_dim']
             return torch.zeros((x.shape[0], output_dim), device=x.device, dtype=x.dtype)
        elif isinstance(self.net, nn.Identity):
             output_dim = config['model_params']['pheno_embedding_dim']
             return torch.zeros((x.shape[0], output_dim), device=x.device, dtype=x.dtype)
        return self.net(x)

# --- GNNClassifier ---
class GNNClassifier(nn.Module):
    """GATv2 based Graph Classifier."""
    def __init__(self, config, node_feature_dim):
        super(GNNClassifier, self).__init__()
        model_cfg = config['model_params']
        gnn_hidden_dim = model_cfg['gnn_hidden_dim']
        out_dim = model_cfg['gnn_output_dim']
        heads = model_cfg['gnn_heads']
        dropout = model_cfg['dropout_rate']

        self.conv1 = GATv2Conv(node_feature_dim, gnn_hidden_dim, heads=heads, dropout=dropout, add_self_loops=True, concat=True)
        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim * heads)
        self.conv2 = GATv2Conv(gnn_hidden_dim * heads, gnn_hidden_dim, heads=heads, dropout=dropout, concat=False, add_self_loops=True)
        self.bn2 = nn.BatchNorm1d(gnn_hidden_dim)

        self.classifier_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(gnn_hidden_dim // 2, out_dim)
        )

    def forward(self, x, edge_index):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaNs or Infs detected in input features to GNNClassifier.")
            x = torch.nan_to_num(x)

        x = self.conv1(x, edge_index)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaNs or Infs after GATv2Conv1.")
            x = torch.nan_to_num(x)

        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.conv1.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaNs or Infs after GATv2Conv2.")
            x = torch.nan_to_num(x)

        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.conv2.dropout, training=self.training)

        x = self.classifier_head(x)
        return x