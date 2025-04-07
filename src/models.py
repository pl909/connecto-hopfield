import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np

# Try to import HopfieldCore from different possible locations
try:
    from hflayers import HopfieldCore
    logging.info("Successfully imported HopfieldCore from hflayers package")
except ImportError:
    try:
        from src.hflayers import HopfieldCore
        logging.info("Successfully imported HopfieldCore from src.hflayers")
    except ImportError:
        try:
            import sys
            import os
            # Add parent directory to path
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from hflayers import HopfieldCore
            logging.info("Successfully imported HopfieldCore after path adjustment")
        except ImportError:
            logging.error("Failed to import HopfieldCore. Make sure the hflayers directory is in your Python path.")
            raise

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings to provide time information.
    
    Follows the original implementation from "Attention Is All You Need".
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: Hidden dimensionality of the input
            max_len: Maximum length of the input sequences
            dropout: Dropout value
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and moved to device with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ACHNN(nn.Module):
    """
    Attentive Connectome-based Hopfield Network for dynamic pain state classification.
    
    Architecture:
    1. Linear embedding layer: Projects 122 brain regions to hidden dimension
    2. Positional encoding: Adds sequence position information
    3. Transformer encoder layers: Process the dynamic information via self-attention 
    4. Hopfield core layer: Learns associative memory patterns that represent brain states
    5. Classification head: Maps retrieved patterns to pain condition classes
    
    Based on:
    - "Hopfield Networks is All You Need" (Ramsauer et al., 2020)
    - "Attention Is All You Need" (Vaswani et al., 2017)
    """
    def __init__(self, config, num_classes=None):
        """
        Args:
            config: Configuration dictionary containing model parameters
            num_classes: Number of output classes (if None, uses config value)
        """
        super().__init__()
        
        model_config = config['achnn_model']
        
        # Model dimensions
        self.input_dim = config['data']['num_regions']  # BASC 122-region atlas
        self.hidden_dim = model_config['hidden_dim']
        
        # Output dimension (number of classes)
        if num_classes is not None:
            self.num_classes = num_classes
        elif 'num_classes' in model_config:
            self.num_classes = model_config['num_classes']
        else:
            raise ValueError("num_classes must be provided either in config or as an argument")
        
        # Dropout rates
        self.embedding_dropout = model_config.get('embedding_dropout', 0.1)
        self.encoder_dropout = model_config.get('encoder_dropout', 0.1)
        self.classifier_dropout = model_config.get('classifier_dropout', 0.1)
        
        # Transformer encoder parameters
        self.num_encoder_layers = model_config.get('num_encoder_layers', 1)
        self.num_self_attn_heads = model_config.get('num_self_attn_heads', 4)
        self.transformer_ff_dim = model_config.get('transformer_ff_dim', self.hidden_dim * 4)
        
        # Hopfield layer parameters
        self.hopfield_num_heads = model_config.get('hopfield_num_heads', 1)
        self.hopfield_num_stored_patterns = model_config.get('hopfield_num_stored_patterns', 10)
        self.hopfield_scaling = model_config.get('hopfield_scaling', 1.0)
        self.hopfield_pattern_dim = model_config.get('hopfield_pattern_dim', self.hidden_dim)
        self.hopfield_update_steps = model_config.get('hopfield_update_steps', 1)
        self.hopfield_update_steps_eps = model_config.get('hopfield_update_steps_eps', 1e-4)
        
        # Use positional encoding?
        self.use_positional_encoding = model_config.get('use_positional_encoding', True)
        
        # Configuration flags for advanced options
        self.return_attention = model_config.get('return_attention', True)
        self.query_selection_method = model_config.get('query_selection_method', 'last')
        self.normalize_patterns = model_config.get('normalize_patterns', False)
        
        # Build model components
        
        # Initial embedding layer: project from region timeseries to hidden dimension
        self.embed = nn.Linear(self.input_dim, self.hidden_dim)
        self.embed_dropout = nn.Dropout(self.embedding_dropout)
        
        # Optional positional encoding
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                self.hidden_dim, 
                max_len=config['data']['seq_len'],
                dropout=model_config.get('pos_encoding_dropout', 0.1)
            )
        
        # Transformer encoder layers with multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_self_attn_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=self.encoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # First normalization layer before Hopfield
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        
        # Hopfield core layer (modern continuous Hopfield network)
        self.hopfield = HopfieldCore(
            embed_dim=self.hidden_dim,
            num_heads=self.hopfield_num_heads,
            dropout=model_config.get('hopfield_dropout', 0.1),
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,  # Same as embed_dim
            vdim=None,  # Same as embed_dim
            head_dim=self.hidden_dim // self.hopfield_num_heads,  # Dimension of each attention head
            pattern_dim=self.hopfield_pattern_dim,  # Dimension of stored patterns
            out_dim=self.hopfield_pattern_dim,  # Output dimension after Hopfield layer
            disable_out_projection=False,
            key_as_static=False,  # Don't use static keys (allow learning)
            query_as_static=False,  # Don't use static queries
            value_as_static=False,  # Don't use static values
            value_as_connected=False,  # Values are not connected to keys
            normalize_pattern=self.normalize_patterns,
            normalize_pattern_affine=self.normalize_patterns,
            normalize_pattern_eps=1e-5,
        )
        
        # Second normalization layer after Hopfield
        self.norm2 = nn.LayerNorm(self.hopfield_pattern_dim)
        
        # Classifier head: project from hopfield output to class logits
        self.classifier = nn.Sequential(
            nn.Dropout(self.classifier_dropout),
            nn.Linear(self.hopfield_pattern_dim, self.num_classes)
        )
        
        # For keeping track of transformer self-attention
        self.self_attn_weights = None
        
        # Initialize parameters with small values to improve training stability
        self._initialize_parameters()
        
        logging.info(f"Initialized ACHNN model with:")
        logging.info(f"  - Input dimension: {self.input_dim}")
        logging.info(f"  - Hidden dimension: {self.hidden_dim}")
        logging.info(f"  - Transformer encoder: {self.num_encoder_layers} layers with {self.num_self_attn_heads} heads")
        logging.info(f"  - Hopfield layer: {self.hopfield_num_stored_patterns} stored patterns, scaling {self.hopfield_scaling}")
        logging.info(f"  - Hopfield parameters: pattern_dim={self.hopfield_pattern_dim}, update_steps={self.hopfield_update_steps}")
        logging.info(f"  - Output classes: {self.num_classes}")
        
    def _initialize_parameters(self):
        """Initialize model parameters for improved training stability."""
        # Initialize embedding layer with small values
        nn.init.xavier_uniform_(self.embed.weight, gain=0.01)
        if self.embed.bias is not None:
            nn.init.zeros_(self.embed.bias)
        
        # Initialize classifier layer
        nn.init.xavier_uniform_(self.classifier[1].weight, gain=0.01)
        nn.init.zeros_(self.classifier[1].bias)
        
    def _select_query_vector(self, x):
        """
        Select query vector from transformer encoder output based on configured method.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        if self.query_selection_method == 'last':
            # Use last time step as query
            return x[:, -1, :]
        elif self.query_selection_method == 'mean':
            # Use mean across time steps as query
            return torch.mean(x, dim=1)
        elif self.query_selection_method == 'first':
            # Use first time step as query
            return x[:, 0, :]
        else:
            logging.warning(f"Unknown query selection method '{self.query_selection_method}', defaulting to 'last'")
            return x[:, -1, :]
    
    def forward(self, x, return_intermediates=False):
        """
        Forward pass through the ACHNN model.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, num_regions]
            return_intermediates: Whether to return intermediate representations for analysis
            
        Returns:
            tuple: (output_logits, hopfield_attention, intermediates) where:
                - output_logits: Tensor of shape [batch_size, num_classes]
                - hopfield_attention: Tensor of shape [batch_size, hopfield_num_stored_patterns] or None
                - intermediates: Dictionary of intermediate representations or None
        """
        batch_size, seq_len, _ = x.shape
        intermediates = {} if return_intermediates else None
        
        # Initial embedding (from regional timeseries to hidden dimension)
        x = self.embed(x)
        x = self.embed_dropout(x)
        
        # Optional positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        if return_intermediates:
            intermediates['embedded'] = x.detach().clone()
            
        # Transformer encoder with self-attention for temporal processing
        x = self.transformer_encoder(x)
        
        if return_intermediates:
            intermediates['transformer_output'] = x.detach().clone()
        
        # Select query vector (typically the last time step)
        query = self._select_query_vector(x)
        
        if return_intermediates:
            intermediates['query'] = query.detach().clone()
            
        # Apply normalization
        query = self.norm1(query)
        
        # Hopfield core layer for pattern retrieval
        # We use the key and value to be the same as query for associative memory retrieval
        # The Hopfield layer will use its learned stored patterns to match against the query
        hopfield_output, attn_weights, raw_assoc, _ = self.hopfield(
            query=query,
            key=query,
            value=query,
            need_weights=True,
            attn_mask=None,  # No attention mask needed
            key_padding_mask=None,  # No padding mask needed
            scaling=self.hopfield_scaling,  # Beta parameter controlling temperature
            update_steps_max=self.hopfield_update_steps,  # Maximum number of pattern retrieval iterations
            update_steps_eps=self.hopfield_update_steps_eps,  # Convergence threshold
            return_raw_associations=True,  # Get raw association scores for analysis
            return_pattern_projections=False  # We don't need pattern projections
        )
        
        # Extract attention weights over stored patterns (useful for analysis)
        if raw_assoc is not None:
            # raw_assoc has shape [batch_size, num_heads, 1, hopfield_num_stored_patterns]
            hopfield_attention = raw_assoc.squeeze(2)  # Remove seq_len dimension (always 1 for query)
        else:
            hopfield_attention = None
            
        if return_intermediates:
            intermediates['hopfield_output'] = hopfield_output.detach().clone()
            if hopfield_attention is not None:
                intermediates['hopfield_attention'] = hopfield_attention.detach().clone()
        
        # Apply second normalization
        hopfield_output = self.norm2(hopfield_output)
        
        # Classification head
        logits = self.classifier(hopfield_output)
        
        if return_intermediates:
            intermediates['pre_logits'] = hopfield_output.detach().clone()
            
        return logits, hopfield_attention, intermediates
    
    def extract_latent_representations(self, dataloader, device='cpu'):
        """
        Extract latent representations from the Hopfield layer for visualization.
        
        Args:
            dataloader: DataLoader with input data
            device: Device to run the model on
            
        Returns:
            tuple: (latent_vectors, labels) for visualization with PCA/t-SNE
        """
        self.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                _, _, intermediates = self.forward(inputs, return_intermediates=True)
                latent_vectors.append(intermediates['pre_logits'].cpu().numpy())
                labels.append(targets.numpy())
                
        return np.vstack(latent_vectors), np.concatenate(labels)
    
    def get_stored_patterns(self):
        """
        Returns the learned stored patterns from the Hopfield layer.
        
        Returns:
            torch.Tensor: Stored pattern weights
        """
        # Access the stored patterns (keys and values) from the Hopfield layer
        # The exact attribute names may vary depending on the HopfieldCore implementation
        # In the provided implementation, stored patterns are part of the Hopfield layer parameters
        
        # This is an approximate implementation - adjust based on actual HopfieldCore internals
        try:
            # Check if we can extract from k_proj_weight of the hopfield layer
            if hasattr(self.hopfield, 'k_proj_weight') and self.hopfield.k_proj_weight is not None:
                return self.hopfield.k_proj_weight.data
            # Alternative: check if it's in the in_proj_weight
            elif hasattr(self.hopfield, 'in_proj_weight') and self.hopfield.in_proj_weight is not None:
                # Assuming the first part corresponds to the query, second to key, third to value
                # Divide by 3 as in_proj_weight contains q, k, v projections
                pattern_size = self.hopfield.in_proj_weight.size(0) // 3
                return self.hopfield.in_proj_weight[pattern_size:2*pattern_size].data
        except (AttributeError, IndexError) as e:
            logging.warning(f"Could not extract stored patterns from Hopfield layer: {e}")
            return None 