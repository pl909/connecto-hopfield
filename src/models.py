import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from torch import Tensor
from typing import Dict, Optional, Tuple, Union, List
import torch.optim as optim
import os
import re
from collections import OrderedDict

# Import Hopfield layer with fallbacks for different execution contexts
try:
    from hflayers import Hopfield
except ImportError:
    try:
        from src.hflayers import Hopfield
    except ImportError:
        try:
            # Try relative import
            from .hflayers import Hopfield
        except ImportError:
            # Try one more approach with direct path
            import sys
            hflayers_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hflayers')
            sys.path.append(hflayers_path)
            try:
                from hopfield import Hopfield
            except ImportError:
                logging.error(f"Could not import Hopfield. Searched in {hflayers_path}")
                raise

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
    def __init__(self, config, num_classes=2):
        """
        Args:
            config: Configuration dictionary containing model parameters
            num_classes: Number of output classes (if None, uses config value)
        """
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Feature normalization
        self.norm = nn.LayerNorm(config['data']['num_regions'])
        
        # Embedding layer with increased dropout
        self.embed = nn.Sequential(
            nn.Linear(config['data']['num_regions'], config['achnn_model']['hidden_dim']),
            nn.LayerNorm(config['achnn_model']['hidden_dim']),
            nn.Dropout(config['achnn_model'].get('embedding_dropout', 0.2)),
            nn.GELU()
        )
        
        # Apply better weight initialization to speed up training
        self._init_weights(self.embed[0])
        
        # Positional encoding with increased dropout
        if config['achnn_model'].get('use_positional_encoding', True):
            self.pos_encoding = nn.Parameter(
                torch.randn(1, config['data']['seq_len'], config['achnn_model']['hidden_dim']) * 0.02  # Scale down init for stability
            )
            self.pos_dropout = nn.Dropout(config['achnn_model'].get('encoder_dropout', 0.2))
        else:
            self.pos_encoding = None
            self.pos_dropout = None
        
        # Transformer encoder with increased dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['achnn_model']['hidden_dim'],
            nhead=config['achnn_model'].get('num_self_attn_heads', 8),
            dim_feedforward=config['achnn_model'].get('transformer_ff_dim', 512),
            dropout=config['achnn_model'].get('encoder_dropout', 0.2),
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['achnn_model']['num_encoder_layers']
        )
        
        # LayerNorm scales outputs to have mean 0 and std 1 for better gradient flow
        self.norm1 = nn.LayerNorm(config['achnn_model']['hidden_dim'])
        self.dropout1 = nn.Dropout(config['achnn_model'].get('dropout', 0.3))
        
        # Hopfield layer with improved stability
        self.hopfield = Hopfield(
            input_size=config['achnn_model']['hidden_dim'],
            hidden_size=config['achnn_model'].get('hopfield_pattern_dim', 128),
            output_size=config['achnn_model']['hidden_dim'],
            num_heads=config['achnn_model'].get('hopfield_num_heads', 2),
            scaling=config['achnn_model'].get('hopfield_scaling', 2.0),
            update_steps_max=config['achnn_model'].get('hopfield_update_steps', 3),
            update_steps_eps=1e-4,
            normalize_stored_pattern=config['achnn_model'].get('normalize_stored_patterns', True),
            normalize_stored_pattern_affine=True,
            normalize_stored_pattern_eps=1e-5,
            normalize_state_pattern=config['achnn_model'].get('normalize_state', True),
            normalize_state_pattern_affine=True,
            normalize_state_pattern_eps=1e-5,
            normalize_pattern_projection=config['achnn_model'].get('normalize_pattern_projection', True),
            normalize_pattern_projection_affine=True,
            normalize_pattern_projection_eps=1e-5,
            normalize_hopfield_space=True,
            normalize_hopfield_space_affine=True,
            normalize_hopfield_space_eps=1e-5,
            dropout=config['achnn_model'].get('dropout', 0.3),
            batch_first=True
        )
        
        # Final classifier with increased dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(config['achnn_model']['hidden_dim']),
            nn.Dropout(config['achnn_model'].get('classifier_dropout', 0.3)),
            nn.Linear(config['achnn_model']['hidden_dim'], num_classes)
        )
        
        # Initialize classifier with better weights
        self._init_weights(self.classifier[-1])
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Log model architecture details
        logging.info("=" * 50)
        logging.info("ACHNN Model Architecture:")
        logging.info("-" * 50)
        logging.info(f"Input dimension: {config['data']['num_regions']}")
        logging.info(f"Hidden dimension: {config['achnn_model']['hidden_dim']}")
        logging.info(f"Sequence length: {config['data']['seq_len']}")
        logging.info(f"Number of classes: {num_classes}")
        logging.info("\nTransformer Configuration:")
        logging.info(f"  - Encoder layers: {config['achnn_model']['num_encoder_layers']}")
        logging.info(f"  - Self-attention heads: {config['achnn_model'].get('num_self_attn_heads', 8)}")
        logging.info(f"  - Feedforward dim: {config['achnn_model'].get('transformer_ff_dim', 512)}")
        logging.info("\nHopfield Layer Configuration:")
        logging.info(f"  - Pattern dimension: {config['achnn_model'].get('hopfield_pattern_dim', 128)}")
        logging.info(f"  - Number of heads: {config['achnn_model'].get('hopfield_num_heads', 2)}")
        logging.info(f"  - Update steps: {config['achnn_model'].get('hopfield_update_steps', 3)}")
        logging.info(f"  - Scaling: {config['achnn_model'].get('hopfield_scaling', 2.0)}")
        logging.info("\nDropout Configuration:")
        logging.info(f"  - Embedding dropout: {config['achnn_model'].get('embedding_dropout', 0.2)}")
        logging.info(f"  - Encoder dropout: {config['achnn_model'].get('encoder_dropout', 0.2)}")
        logging.info(f"  - Classifier dropout: {config['achnn_model'].get('classifier_dropout', 0.3)}")
        logging.info("\nModel Size:")
        logging.info(f"  - Total parameters: {total_params:,}")
        logging.info(f"  - Trainable parameters: {trainable_params:,}")
        logging.info("=" * 50)
        
    def _init_weights(self, module):
        """Initialize the weights - this improves convergence speed significantly"""
        if isinstance(module, nn.Linear):
            # Slightly better than PyTorch default - scales with hidden size
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x, return_intermediates=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            return_intermediates: Whether to return intermediate activations
            
        Returns:
            tuple: (class_logits, hopfield_attention, intermediate_activations)
            - class_logits: Output class logits of shape [batch_size, num_classes]
            - hopfield_attention: Attention weights from Hopfield layer
            - intermediate_activations: Dict of intermediate activations (if return_intermediates=True)
        """
        intermediates = {} if return_intermediates else None
        batch_size, seq_len, num_features = x.shape
        
        # Skip some operations during evaluation to save memory and time
        if not self.training and not return_intermediates:
            # Fast path for evaluation - normalize in one step
            with torch.no_grad():  # Extra safety with no_grad
                # Inline the normalization for speed
                x = self.norm(x.reshape(-1, num_features)).reshape(batch_size, seq_len, num_features)
                x = self.embed(x)
                
                # Add positional encoding if available
                if self.pos_encoding is not None:
                    x = x + self.pos_encoding
                    x = self.pos_dropout(x)
                
                # Process through transformer 
                x = self.transformer_encoder(x)
                x = self.norm1(x)
                
                # Select last token as query and immediately free memory
                hopfield_query = x[:, -1, :].contiguous()
                del x
                # Check device using a tensor that still exists
                torch.cuda.empty_cache() if hopfield_query.device.type == 'cuda' else None
                
                # Simple residual connection for speed
                transformer_features = hopfield_query.clone()
                
                # Reshape query for hopfield (batch_size, 1, hidden_dim)
                hopfield_query = hopfield_query.unsqueeze(1)
                
                try:
                    # Fast forward pass through Hopfield during evaluation
                    # Use inference_mode for fastest processing
                    hopfield_output = self.hopfield(
                        input=(hopfield_query, hopfield_query, hopfield_query)
                    )
                    
                    # Process output and free memory immediately
                    retrieved = hopfield_output.squeeze(1)
                    del hopfield_output, hopfield_query
                    torch.cuda.empty_cache() if transformer_features.device.type == 'cuda' else None
                    
                except RuntimeError as e:
                    print(f"Hopfield layer error during evaluation: {e}")
                    retrieved = transformer_features
                    
                # Add residual connection
                retrieved = retrieved + transformer_features
                del transformer_features
                
                # Apply final normalization and classifier
                retrieved = self.norm1(retrieved)
                logits = self.classifier(retrieved)
                del retrieved
                
                return logits
        
        # Full path for training with all operations
        
        # Normalize features for each region independently
        # Reshape to [batch_size * seq_len, num_features]
        x_flat = x.reshape(-1, num_features)
        x_norm = self.norm(x_flat)
        # Reshape back to [batch_size, seq_len, num_features]
        x = x_norm.reshape(batch_size, seq_len, num_features)
        
        # Free memory
        del x_flat, x_norm
        
        # Initial projection to hidden dimension
        x = self.embed(x)  # [batch_size, seq_len, hidden_dim]
        if return_intermediates:
            intermediates['embed_output'] = x.detach().clone()
        
        # Add positional encoding if enabled
        if self.pos_encoding is not None:
            x = x + self.pos_encoding
            x = self.pos_dropout(x)
            if return_intermediates:
                intermediates['pos_encoded'] = x.detach().clone()
        
        # Pass through transformer encoder layers
        x = self.transformer_encoder(x)
        if return_intermediates:
            intermediates['transformer_output'] = x.detach().clone()
            
        # Apply normalization
        x = self.norm1(x)
        x = self.dropout1(x)
            
        # Select query vector for Hopfield retrieval based on config
        query_method = self.config['achnn_model'].get('query_selection_method', 'last') # Default to last if not specified
        if query_method == 'mean':
            hopfield_query = x.mean(dim=1) # Mean across sequence length
        elif query_method == 'last':
            hopfield_query = x[:, -1, :] # Last token output
        else:
            logging.warning(f"Unknown query_selection_method '{query_method}'. Defaulting to 'last'.")
            hopfield_query = x[:, -1, :]
        
        # Free memory - we don't need the full sequence output anymore if using mean/last
        del x
        
        if return_intermediates:
            intermediates['hopfield_query'] = hopfield_query.detach().clone()
        
        # Stabilize inputs with normalization (removed for direct path)
        # hopfield_query = F.normalize(hopfield_query, p=2, dim=1) * 3.0
        
        # --- Option: Bypass Hopfield --- 
        # Directly use the features from the transformer/pooling step
        retrieved = hopfield_query # Use the pooled query directly
        hopfield_attn = None # No attention from Hopfield
        
        # --- Original Hopfield Path (Commented Out) ---
        # transformer_features = hopfield_query.clone()
        # hopfield_query = hopfield_query.unsqueeze(1)
        # orig_dtype = hopfield_query.dtype
        # hopfield_query_float = hopfield_query.to(torch.float32) if orig_dtype != torch.float32 else hopfield_query
        # hopfield_attn = None
        # try:
        #     hopfield_output = self.hopfield(
        #         input=(hopfield_query_float, hopfield_query_float, hopfield_query_float)
        #     )
        #     retrieved = hopfield_output.squeeze(1)
        #     if return_intermediates:
        #         # Create a dummy attention tensor 
        #         batch_size = retrieved.size(0)
        #         # The shape depends on hopfield config, assume [batch, heads, patterns] -> [batch, 1, 1] after squeeze/mean
        #         num_patterns = self.config['achnn_model'].get('hopfield_num_stored_patterns', 1) # Get num_patterns
        #         num_heads_hf = self.config['achnn_model'].get('hopfield_num_heads', 1)
        #         # Placeholder shape, adjust if needed based on actual internal attention access
        #         hopfield_attn = torch.ones((batch_size, num_heads_hf, 1), device=retrieved.device) 
        #     retrieved = retrieved * self.config['achnn_model']['hopfield_scaling'] # Use hopfield_scaling
        #     retrieved = retrieved + transformer_features
        #     del hopfield_output, hopfield_query, hopfield_query_float
        # except RuntimeError as e:
        #     print(f"Hopfield layer error: {e}")
        #     retrieved = transformer_features
        #     hopfield_attn = None
        # del transformer_features
        # --------------------------------------
        
        if return_intermediates:
            # Store the output before the final classifier, which represents the 'latent' state now
            intermediates['hopfield_output'] = retrieved.detach().clone()
            
        # Apply second normalization using the same layer norm as before
        retrieved = self.norm1(retrieved)
        
        # Pass through classifier to get class logits
        logits = self.classifier(retrieved)
        
        # Free memory
        del retrieved
        
        if return_intermediates:
            return logits, hopfield_attn, intermediates
        else:
            return logits
    
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
                try:
                    # Explicitly set return_intermediates to True to get intermediate representations
                    forward_output = self.forward(inputs, return_intermediates=True)
                    
                    # Extract the intermediates from the returned tuple
                    if isinstance(forward_output, tuple) and len(forward_output) == 3:
                        _, _, intermediates = forward_output
                        if 'hopfield_output' in intermediates:
                            latent_vectors.append(intermediates['hopfield_output'].cpu().numpy())
                        else:
                            print("Warning: 'hopfield_output' not found in intermediates dictionary")
                            continue
                    else:
                        # If we have a problem with the forward method, log it and skip this batch
                        print(f"Warning: forward pass did not return expected intermediates. Got: {type(forward_output)}")
                        continue
                except Exception as e:
                    print(f"Error during latent vector extraction: {e}")
                    continue
                
                labels.append(targets.numpy())
                
        if len(latent_vectors) == 0:
            raise RuntimeError("No latent vectors were collected. Check the forward pass.")
            
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

def initialize_model(config, task_type='classification'):
    """
    Initialize a model based on the given configuration.
    
    Args:
        config: Configuration dictionary
        task_type: Type of task (classification or regression)
        
    Returns:
        model: Initialized model
    """
    # Extract model parameters from config
    embedding_dim = config.get('model', {}).get('embedding_dim', 128)
    hidden_dim = config.get('model', {}).get('hidden_dim', 256)
    num_layers = config.get('model', {}).get('num_encoder_layers', 2)
    num_heads = config.get('model', {}).get('num_attention_heads', 4)
    dropout = config.get('model', {}).get('dropout', 0.3)
    embedding_dropout = config.get('model', {}).get('embedding_dropout', 0.2)
    encoder_dropout = config.get('model', {}).get('encoder_dropout', 0.2)
    classifier_dropout = config.get('model', {}).get('classifier_dropout', 0.3)
    use_positional_encoding = config.get('model', {}).get('use_positional_encoding', True)
    hopfield_beta = config.get('model', {}).get('hopfield_beta', 1.0)
    hopfield_alpha = config.get('model', {}).get('hopfield_alpha', 1.0)
    
    # Get sequence length from data section
    sequence_length = config.get('data', {}).get('sequence_length')
    if not sequence_length:
        sequence_length = config.get('data', {}).get('seq_len', 100)
    
    # Determine number of regions
    n_regions = config.get('data', {}).get('num_regions', 116)
    
    # For regression, set output_size to 1, for classification use num_classes or default to 2
    if task_type == 'regression':
        output_size = 1
        logging.info("Initializing model for regression task (output_size=1)")
    else:
        output_size = config.get('model', {}).get('num_classes', 2)
        logging.info(f"Initializing model for classification task (output_size={output_size})")
    
    # Initialize model
    model = AttentionConnectoHopfield(
        n_regions=n_regions,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_size=output_size,
        sequence_length=sequence_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        embedding_dropout=embedding_dropout,
        encoder_dropout=encoder_dropout,
        classifier_dropout=classifier_dropout,
        use_positional_encoding=use_positional_encoding,
        hopfield_beta=hopfield_beta,
        hopfield_alpha=hopfield_alpha,
        regression=(task_type == 'regression')
    )
    
    # Log model details
    logging.info(f"Initialized {task_type} model with:")
    logging.info(f"  - {n_regions} regions")
    logging.info(f"  - Embedding dimension: {embedding_dim}")
    logging.info(f"  - Hidden dimension: {hidden_dim}")
    logging.info(f"  - Output size: {output_size}")
    logging.info(f"  - Sequence length: {sequence_length}")
    logging.info(f"  - Number of layers: {num_layers}")
    logging.info(f"  - Number of attention heads: {num_heads}")
    
    return model

class AttentionConnectoHopfield(nn.Module):
    def __init__(self, n_regions, embedding_dim, hidden_dim, output_size=2, sequence_length=100,
                num_layers=1, num_heads=4, dropout=0.1, embedding_dropout=0.1,
                encoder_dropout=0.1, classifier_dropout=0.2, use_positional_encoding=True,
                hopfield_beta=1.0, hopfield_alpha=0.5, regression=False):
        """
        Initialize the Attention Connectome model with Hopfield layer.
        
        Args:
            n_regions: Number of brain regions
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layer
            output_size: Size of the output (num_classes for classification, 1 for regression)
            sequence_length: Length of sequence
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            embedding_dropout: Dropout rate for embeddings
            encoder_dropout: Dropout rate for encoder
            classifier_dropout: Dropout rate for classifier
            use_positional_encoding: Whether to use positional encoding
            hopfield_beta: Beta parameter for Hopfield layer
            hopfield_alpha: Alpha parameter for Hopfield layer
            regression: Whether this is a regression task
        """
        super().__init__()
        
        self.n_regions = n_regions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.embedding_dropout_rate = embedding_dropout
        self.encoder_dropout_rate = encoder_dropout
        self.classifier_dropout_rate = classifier_dropout
        self.use_positional_encoding = use_positional_encoding
        self.hopfield_beta = hopfield_beta
        self.hopfield_alpha = hopfield_alpha
        self.regression = regression
        
        # Logging the model settings
        logging.debug(f"Initializing AttentionConnectoHopfield with {n_regions} regions, "
                    f"embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
                    f"output_size={output_size}, task_type={'regression' if regression else 'classification'}")
        
        # Region embeddings
        self.region_embeddings = nn.Parameter(torch.randn(n_regions, embedding_dim))
        self.norm_layer1 = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(embedding_dim, max_len=sequence_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=encoder_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.norm_layer2 = nn.LayerNorm(embedding_dim)
        
        # Hopfield layer
        self.hopfield = Hopfield(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            output_size=embedding_dim,
            num_heads=1,
            scaling=hopfield_beta
        )
        
        # Classification head
        if regression:
            # For regression, use a single output neuron
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # For classification, output logits for each class
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, output_size)
            )
    
    def forward(self, x, return_intermediates=False):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_regions)
            return_intermediates: Whether to return intermediate activations
            
        Returns:
            logits: Output logits of shape (batch_size, output_size) for classification
                   or (batch_size, 1) for regression
            attn_weights: Attention weights if return_intermediates is True
            intermediates: Dictionary of intermediate activations if return_intermediates is True
        """
        batch_size, seq_len, n_features = x.shape
        
        try:
            # First apply embeddings directly without flattening
            # Apply embedding by matrix multiplication with region embeddings
            x_embedded = torch.matmul(x, self.region_embeddings)  # Shape: [batch_size, seq_len, embedding_dim]
            
            # Now normalize along the embedding dimension
            x_embedded = self.norm_layer1(x_embedded)  # Shape: [batch_size, seq_len, embedding_dim]
            x_embedded = self.embedding_dropout(x_embedded)
            
            if self.use_positional_encoding:
                x_embedded = self.positional_encoding(x_embedded)
            
            # Pass through transformer encoder
            transformer_out = self.transformer_encoder(x_embedded)
            transformer_out = self.norm_layer2(transformer_out)
            
            # Get query vector (average over sequence dimension)
            query_vector = torch.mean(transformer_out, dim=1)
            
            # Ensure query has the right dtype for numerical stability
            orig_dtype = query_vector.dtype
            if orig_dtype != torch.float32:
                query_vector = query_vector.to(torch.float32)
            
            # Get stored pattern from transformer output
            try:
                # Ensure key and value have compatible dimensions with query
                # query has shape [batch_size, 1, hidden_dim]
                # Create key and value with same sequence length as query for compatibility
                key_value = query_vector.unsqueeze(1)  # Use the same tensor for key and value
                
                # Hopfield forward returns only the output tensor, not attention weights
                hopfield_output = self.hopfield(
                    input=(query_vector.unsqueeze(1), key_value, key_value),
                    stored_pattern_padding_mask=None,
                    association_mask=None
                )
                hopfield_output = hopfield_output.squeeze(1)
                
                # If we need attention weights and return_intermediates is True, get them separately
                attn_weights = None
                if return_intermediates:
                    try:
                        # Use get_association_matrix to get attention weights if needed
                        with torch.no_grad():
                            attn_weights = self.hopfield.get_association_matrix(
                                input=(query_vector.unsqueeze(1), key_value, key_value)
                            )
                    except Exception as e:
                        logging.warning(f"Could not get attention weights: {str(e)}")
                
                # Convert back to original dtype if needed
                if orig_dtype != torch.float32:
                    hopfield_output = hopfield_output.to(orig_dtype)
                
            except RuntimeError as e:
                logging.warning(f"Error in Hopfield layer: {str(e)}. Using transformer features instead.")
                hopfield_output = query_vector
                attn_weights = None
            
            # Apply classifier
            classifier_output = self.classifier(hopfield_output)
            
            # For regression models, squeeze the output to match the target shape
            if self.regression:
                classifier_output = classifier_output.squeeze(-1)
            
            # Return appropriate outputs based on return_intermediates flag
            if return_intermediates:
                intermediates = {
                    'transformer_output': transformer_out,
                    'query_vector': query_vector,
                    'hopfield_output': hopfield_output
                }
                return classifier_output, attn_weights, intermediates
            else:
                return classifier_output
        except RuntimeError as e:
            logging.error(f"Runtime error in forward pass: {str(e)}")
            logging.error(f"Input shape: {x.shape}, features: {n_features}, embedding_dim: {self.embedding_dim}")
            raise 