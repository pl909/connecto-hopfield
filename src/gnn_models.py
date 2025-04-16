import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import logging
from typing import List, Dict, Optional, Union, Tuple, Callable

logger = logging.getLogger(__name__)


class GNN(nn.Module):
    """Base class for Graph Neural Networks.
    
    This is an abstract base class for graph neural network models. It defines
    the common interface and functionality for different GNN implementations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5, 
                 pool_method: str = 'mean', use_edge_attr: bool = True):
        """Initialize the GNN base class.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes)
            num_layers (int): Number of graph convolutional layers
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'sum', 'max')
            use_edge_attr (bool): Whether to use edge attributes
        """
        super(GNN, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_method = pool_method
        self.use_edge_attr = use_edge_attr
    
    def pool(self, x, batch, method=None):
        """Pool node features to graph-level representations.
        
        Args:
            x (torch.Tensor): Node features
            batch (torch.Tensor): Batch assignment for nodes
            method (str, optional): Pooling method, defaults to self.pool_method
            
        Returns:
            torch.Tensor: Graph-level representations
        """
        method = method or self.pool_method
        
        if method == 'mean':
            return global_mean_pool(x, batch)
        elif method == 'sum':
            return global_add_pool(x, batch)
        elif method == 'max':
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")
    
    def forward(self, data):
        """Forward pass.
        
        This method should be implemented by subclasses.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Class logits
        """
        raise NotImplementedError("Subclasses must implement forward method")


class GCN(GNN):
    """Graph Convolutional Network (GCN).
    
    This model implements the Graph Convolutional Network as described in the paper
    "Semi-supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017).
    It applies multiple layers of graph convolutions followed by pooling and classification.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5, 
                 pool_method: str = 'mean', use_edge_attr: bool = True,
                 batch_norm: bool = True):
        """Initialize the GCN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes)
            num_layers (int): Number of graph convolutional layers
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'sum', 'max')
            use_edge_attr (bool): Whether to use edge attributes
            batch_norm (bool): Whether to use batch normalization
        """
        super(GCN, self).__init__(
            input_dim, hidden_dim, output_dim, num_layers, dropout, pool_method, use_edge_attr
        )
        
        # Initialize layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Class logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and self.use_edge_attr else None
        
        # Apply graph convolutions with non-linearity and dropout
        for i, conv in enumerate(self.convs):
            # Apply convolution
            if edge_attr is not None and self.use_edge_attr:
                x = conv(x, edge_index, edge_attr.squeeze(-1))
            else:
                x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply non-linearity and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class GAT(GNN):
    """Graph Attention Network (GAT).
    
    This model implements the Graph Attention Network as described in the paper
    "Graph Attention Networks" (Veličković et al., 2018).
    It applies multiple layers of graph attention followed by pooling and classification.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5, 
                 pool_method: str = 'mean', use_edge_attr: bool = True,
                 batch_norm: bool = True, heads: int = 4, concat: bool = True):
        """Initialize the GAT model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes)
            num_layers (int): Number of graph attention layers
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'sum', 'max')
            use_edge_attr (bool): Whether to use edge attributes
            batch_norm (bool): Whether to use batch normalization
            heads (int): Number of attention heads
            concat (bool): Whether to concatenate or average attention heads
        """
        super(GAT, self).__init__(
            input_dim, hidden_dim, output_dim, num_layers, dropout, pool_method, use_edge_attr
        )
        
        # Store additional parameters
        self.heads = heads
        self.concat = concat
        
        # Calculate dimensions for multi-head attention
        # If concat=True, output dim = hidden_dim * heads
        # If concat=False, output dim = hidden_dim
        if concat:
            conv_hidden_dim = hidden_dim // heads
        else:
            conv_hidden_dim = hidden_dim
        
        # Initialize layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # Input layer
        self.convs.append(GATConv(input_dim, conv_hidden_dim, heads=heads, concat=concat, dropout=dropout))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, conv_hidden_dim, heads=heads, concat=concat, dropout=dropout))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Class logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph attention layers with non-linearity and dropout
        for i, conv in enumerate(self.convs):
            # Apply attention
            x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply non-linearity and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class DynamicEdgeGNN(GNN):
    """Dynamic Edge Update Graph Neural Network.
    
    This model implements a GNN that dynamically updates edge features based on
    connected nodes. It uses GCN or SAGE as the core convolution layer and adds
    functionality to update edge features during message passing.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.5, 
                 pool_method: str = 'mean', use_edge_attr: bool = True,
                 batch_norm: bool = True, edge_dim: int = 1,
                 conv_type: str = 'gcn', update_edges: bool = True):
        """Initialize the DynamicEdgeGNN model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension (number of classes)
            num_layers (int): Number of graph convolutional layers
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'sum', 'max')
            use_edge_attr (bool): Whether to use edge attributes
            batch_norm (bool): Whether to use batch normalization
            edge_dim (int): Edge feature dimension
            conv_type (str): Type of convolution ('gcn', 'sage')
            update_edges (bool): Whether to update edge features
        """
        super(DynamicEdgeGNN, self).__init__(
            input_dim, hidden_dim, output_dim, num_layers, dropout, pool_method, use_edge_attr
        )
        
        # Store additional parameters
        self.edge_dim = edge_dim
        self.conv_type = conv_type
        self.update_edges = update_edges
        
        # Initialize node convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # Initialize edge update layers if needed
        if update_edges:
            self.edge_update_mlps = nn.ModuleList()
        
        # Input layer
        if conv_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Edge update for input layer
        if update_edges:
            # MLP to update edge features based on connected nodes
            # Input: source node features + target node features + edge features
            edge_update_mlp = nn.Sequential(
                nn.Linear(input_dim * 2 + edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, edge_dim)
            )
            self.edge_update_mlps.append(edge_update_mlp)
        
        # Hidden layers
        for i in range(num_layers - 1):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Edge update for hidden layer
            if update_edges:
                edge_update_mlp = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, edge_dim)
                )
                self.edge_update_mlps.append(edge_update_mlp)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def update_edge_features(self, x, edge_index, edge_attr, layer_idx):
        """Update edge features based on connected nodes.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            edge_attr (torch.Tensor): Edge features
            layer_idx (int): Current layer index
            
        Returns:
            torch.Tensor: Updated edge features
        """
        # Get source and target node features for each edge
        src, dst = edge_index
        src_features = x[src]
        dst_features = x[dst]
        
        # Concatenate source features, target features, and edge features
        edge_inputs = torch.cat([src_features, dst_features, edge_attr], dim=1)
        
        # Apply MLP to update edge features
        updated_edge_attr = self.edge_update_mlps[layer_idx](edge_inputs)
        
        # Residual connection
        if edge_attr.shape == updated_edge_attr.shape:
            updated_edge_attr = updated_edge_attr + edge_attr
        
        return updated_edge_attr
    
    def forward(self, data):
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Class logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # If no edge attributes but they are required, create zero tensors
        if edge_attr is None and self.use_edge_attr:
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.edge_dim, device=x.device)
        
        # Apply graph convolutions with edge updates
        for i, conv in enumerate(self.convs):
            # Update edge features if enabled
            if self.update_edges and edge_attr is not None:
                edge_attr = self.update_edge_features(x, edge_index, edge_attr, i)
            
            # Apply convolution
            if edge_attr is not None and self.use_edge_attr and self.conv_type == 'gcn':
                x = conv(x, edge_index, edge_attr.squeeze(-1))
            else:
                x = conv(x, edge_index)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply non-linearity and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply global pooling
        x = self.pool(x, batch)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class GCNBrainNetwork(nn.Module):
    """Graph Convolutional Network for brain connectivity data.
    
    This model uses GCN layers to process brain connectivity data
    represented as graphs.
    """
    
    def __init__(self, num_node_features: int, num_classes: int, 
                 hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, pool_method: str = 'mean',
                 batch_norm: bool = True, residual: bool = True):
        """Initialize the GCN model.
        
        Args:
            num_node_features (int): Dimension of input node features
            num_classes (int): Number of output classes
            hidden_channels (int): Dimension of hidden representations
            num_layers (int): Number of GCN layers
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'max', 'add', 'attention')
            batch_norm (bool): Whether to use batch normalization
            residual (bool): Whether to use residual connections
        """
        super(GCNBrainNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.residual = residual
        self.pool_method = pool_method
        
        # Input layer
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(num_node_features, hidden_channels))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        if batch_norm:
            for i in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        else:
            self.batch_norms = None
        
        # Output layer for graph classification
        if pool_method == 'attention':
            # Attention pooling
            self.attention = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, 1)
            )
        
        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
        logger.info(f"Initialized GCN with {num_layers} layers, {hidden_channels} hidden channels")
        logger.info(f"Dropout: {dropout}, Pool method: {pool_method}")
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_latent=False):
        """Forward pass through the GCN model.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features]
            edge_index (Tensor): Graph edge indices of shape [2, num_edges]
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, num_edge_features]
            batch (Tensor): Batch indices of shape [num_nodes]
            return_latent (bool): Whether to return latent representations
            
        Returns:
            Tensor: Logits for each graph, shape [batch_size, num_classes]
            Tensor (optional): Latent representations, only if return_latent=True
        """
        # If batch is not provided, assume all nodes belong to a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial features
        prev_x = None
        
        # Process through GCN layers
        for i, conv in enumerate(self.conv_layers):
            # Store previous features for residual connection
            if self.residual and i > 0:
                prev_x = x
            
            # Apply convolution
            x = conv(x, edge_index, edge_attr)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection if enabled
            if self.residual and i > 0 and x.size() == prev_x.size():
                x = x + prev_x
        
        # Pooling: aggregate node features to graph features
        if self.pool_method == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            pooled = global_max_pool(x, batch)
        elif self.pool_method == 'add':
            pooled = global_add_pool(x, batch)
        elif self.pool_method == 'attention':
            # Attention-based pooling
            attention_scores = self.attention(x).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=0)
            pooled = torch.matmul(attention_weights.unsqueeze(-1).transpose(0, 1), x).squeeze(0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")
        
        # Apply MLP to get final predictions
        logits = self.mlp(pooled)
        
        if return_latent:
            return logits, pooled
        else:
            return logits
    
    def extract_latent_representations(self, data_loader):
        """Extract latent representations from the model.
        
        Args:
            data_loader: PyTorch Geometric DataLoader
            
        Returns:
            tuple: (latent_representations, labels)
        """
        self.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data in data_loader:
                # Move data to device
                if hasattr(data, 'to'):
                    data = data.to(next(self.parameters()).device)
                
                # Extract features
                x, edge_index = data.x, data.edge_index
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                batch = data.batch if hasattr(data, 'batch') else None
                
                # Forward pass with latent representations
                _, latent = self.forward(x, edge_index, edge_attr, batch, return_latent=True)
                
                # Store latent vectors and labels
                latent_vectors.append(latent.cpu())
                if hasattr(data, 'y'):
                    labels.append(data.y.cpu())
        
        # Concatenate batches
        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0) if labels else None
        
        return latent_vectors, labels


class GATBrainNetwork(nn.Module):
    """Graph Attention Network for brain connectivity data.
    
    This model uses GAT layers with multi-head attention to process
    brain connectivity data represented as graphs.
    """
    
    def __init__(self, num_node_features: int, num_classes: int, 
                 hidden_channels: int = 64, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.5, 
                 pool_method: str = 'mean', concat: bool = True):
        """Initialize the GAT model.
        
        Args:
            num_node_features (int): Dimension of input node features
            num_classes (int): Number of output classes
            hidden_channels (int): Dimension of hidden representations
            num_layers (int): Number of GAT layers
            heads (int): Number of attention heads
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'max', 'add')
            concat (bool): Whether to concatenate or average multi-head attention
        """
        super(GATBrainNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.pool_method = pool_method
        
        # Input layer
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(
            num_node_features, hidden_channels, 
            heads=heads, concat=concat, dropout=dropout
        ))
        
        # Determine the output dimension of attention layers
        gat_out_channels = hidden_channels * heads if concat else hidden_channels
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.conv_layers.append(GATConv(
                gat_out_channels, hidden_channels,
                heads=heads, concat=concat, dropout=dropout
            ))
        
        # Final dimension after all attention layers
        final_dim = gat_out_channels
        
        # MLP for final classification
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        
        logger.info(f"Initialized GAT with {num_layers} layers, {hidden_channels} channels, {heads} heads")
        logger.info(f"Dropout: {dropout}, Pool method: {pool_method}")
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_latent=False, return_attention=False):
        """Forward pass through the GAT model.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features]
            edge_index (Tensor): Graph edge indices of shape [2, num_edges]
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, num_edge_features]
            batch (Tensor): Batch indices of shape [num_nodes]
            return_latent (bool): Whether to return latent representations
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Tensor: Logits for each graph, shape [batch_size, num_classes]
            Tensor (optional): Latent representations if return_latent=True
            List (optional): Attention weights if return_attention=True
        """
        # If batch is not provided, assume all nodes belong to a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Store attention weights if requested
        attention_weights = [] if return_attention else None
        
        # Process through GAT layers
        for i, conv in enumerate(self.conv_layers):
            # Apply convolution with attention
            if return_attention:
                x, attention = conv(x, edge_index, edge_attr, return_attention_weights=True)
                attention_weights.append(attention)
            else:
                x = conv(x, edge_index, edge_attr)
            
            # Apply activation and dropout
            if i < self.num_layers - 1:  # No activation on final layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling: aggregate node features to graph features
        if self.pool_method == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            pooled = global_max_pool(x, batch)
        elif self.pool_method == 'add':
            pooled = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")
        
        # Apply MLP to get final predictions
        logits = self.mlp(pooled)
        
        # Return requested outputs
        if return_latent and return_attention:
            return logits, pooled, attention_weights
        elif return_latent:
            return logits, pooled
        elif return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def extract_latent_representations(self, data_loader):
        """Extract latent representations from the model.
        
        Args:
            data_loader: PyTorch Geometric DataLoader
            
        Returns:
            tuple: (latent_representations, labels)
        """
        self.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data in data_loader:
                # Move data to device
                if hasattr(data, 'to'):
                    data = data.to(next(self.parameters()).device)
                
                # Extract features
                x, edge_index = data.x, data.edge_index
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                batch = data.batch if hasattr(data, 'batch') else None
                
                # Forward pass with latent representations
                _, latent = self.forward(x, edge_index, edge_attr, batch, return_latent=True)
                
                # Store latent vectors and labels
                latent_vectors.append(latent.cpu())
                if hasattr(data, 'y'):
                    labels.append(data.y.cpu())
        
        # Concatenate batches
        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0) if labels else None
        
        return latent_vectors, labels


class GNN(nn.Module):
    """Generic Graph Neural Network for brain connectivity data.
    
    This model allows for different types of graph convolution layers
    to be used for processing brain connectivity data.
    """
    
    def __init__(self, num_node_features: int, num_classes: int, 
                 hidden_channels: List[int] = [64, 64], 
                 conv_type: str = 'gcn',
                 dropout: float = 0.5, pool_method: str = 'mean',
                 batch_norm: bool = True, residual: bool = True,
                 activation: str = 'relu', 
                 heads: int = 4, edge_dim: Optional[int] = None):
        """Initialize the GNN model.
        
        Args:
            num_node_features (int): Dimension of input node features
            num_classes (int): Number of output classes
            hidden_channels (list): Dimensions of hidden representations for each layer
            conv_type (str): Type of convolution ('gcn', 'gat', 'graph')
            dropout (float): Dropout probability
            pool_method (str): Pooling method ('mean', 'max', 'add')
            batch_norm (bool): Whether to use batch normalization
            residual (bool): Whether to use residual connections
            activation (str): Activation function ('relu', 'leaky_relu', 'elu')
            heads (int): Number of attention heads (only for GAT)
            edge_dim (int): Dimension of edge features (if used)
        """
        super(GNN, self).__init__()
        
        self.num_layers = len(hidden_channels)
        self.dropout = dropout
        self.residual = residual
        self.pool_method = pool_method
        self.conv_type = conv_type
        self.activation = activation
        
        # Input layer dimensions
        in_channels = num_node_features
        
        # Create convolution layers
        self.conv_layers = nn.ModuleList()
        
        for i, out_channels in enumerate(hidden_channels):
            if conv_type.lower() == 'gcn':
                self.conv_layers.append(GCNConv(in_channels, out_channels))
            elif conv_type.lower() == 'gat':
                # For GAT, the output channels depend on whether we concatenate or average heads
                concat = True  # Default: concatenate attention heads
                self.conv_layers.append(GATConv(
                    in_channels, out_channels // heads if concat else out_channels,
                    heads=heads, concat=concat, dropout=dropout, 
                    edge_dim=edge_dim
                ))
                # Update in_channels based on concatenation or averaging
                in_channels = out_channels if concat else out_channels
                continue
            elif conv_type.lower() == 'graph':
                self.conv_layers.append(GraphConv(
                    in_channels, out_channels, aggr='mean',
                ))
            else:
                raise ValueError(f"Unknown convolution type: {conv_type}")
            
            # Update input dimension for next layer
            in_channels = out_channels
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        if batch_norm:
            for i in range(self.num_layers):
                out_channels = hidden_channels[i]
                if conv_type.lower() == 'gat' and concat:
                    out_channels = hidden_channels[i]
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
        else:
            self.batch_norms = None
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels[-1] // 2, num_classes)
        )
        
        logger.info(f"Initialized {conv_type.upper()} with {self.num_layers} layers")
        logger.info(f"Hidden dimensions: {hidden_channels}")
        if conv_type.lower() == 'gat':
            logger.info(f"Using {heads} attention heads")
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, return_latent=False):
        """Forward pass through the GNN model.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features]
            edge_index (Tensor): Graph edge indices of shape [2, num_edges]
            edge_attr (Tensor): Edge feature matrix of shape [num_edges, num_edge_features]
            batch (Tensor): Batch indices of shape [num_nodes]
            return_latent (bool): Whether to return latent representations
            
        Returns:
            Tensor: Logits for each graph, shape [batch_size, num_classes]
            Tensor (optional): Latent representations, only if return_latent=True
        """
        # If batch is not provided, assume all nodes belong to a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Process through convolution layers
        for i, conv in enumerate(self.conv_layers):
            # Store previous features for residual connection
            prev_x = x if self.residual and i > 0 else None
            
            # Apply convolution based on type
            if self.conv_type.lower() == 'gcn':
                x = conv(x, edge_index)
            elif self.conv_type.lower() == 'gat':
                x = conv(x, edge_index, edge_attr)
            elif self.conv_type.lower() == 'graph':
                x = conv(x, edge_index, edge_attr)
            
            # Apply batch normalization if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation function
            if self.activation == 'relu':
                x = F.relu(x)
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(x, negative_slope=0.2)
            elif self.activation == 'elu':
                x = F.elu(x)
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection if enabled and dimensions match
            if prev_x is not None and prev_x.size() == x.size():
                x = x + prev_x
        
        # Pooling: aggregate node features to graph features
        if self.pool_method == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.pool_method == 'max':
            pooled = global_max_pool(x, batch)
        elif self.pool_method == 'add':
            pooled = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")
        
        # Get final predictions
        logits = self.classifier(pooled)
        
        if return_latent:
            return logits, pooled
        else:
            return logits
    
    def extract_latent_representations(self, data_loader):
        """Extract latent representations from the model.
        
        Args:
            data_loader: PyTorch Geometric DataLoader
            
        Returns:
            tuple: (latent_representations, labels)
        """
        self.eval()
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data in data_loader:
                # Move data to device
                if hasattr(data, 'to'):
                    data = data.to(next(self.parameters()).device)
                
                # Extract features
                x, edge_index = data.x, data.edge_index
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                batch = data.batch if hasattr(data, 'batch') else None
                
                # Forward pass with latent representations
                _, latent = self.forward(x, edge_index, edge_attr, batch, return_latent=True)
                
                # Store latent vectors and labels
                latent_vectors.append(latent.cpu())
                if hasattr(data, 'y'):
                    labels.append(data.y.cpu())
        
        # Concatenate batches
        latent_vectors = torch.cat(latent_vectors, dim=0)
        labels = torch.cat(labels, dim=0) if labels else None
        
        return latent_vectors, labels


def create_gnn_model(model_type: str, input_dim: int, hidden_dim: int, output_dim: int, 
                   model_params: Dict = None) -> nn.Module:
    """Create a GNN model of the specified type.
    
    Args:
        model_type (str): Type of GNN model ('gcn', 'gat', 'dynamic_edge')
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (number of classes)
        model_params (dict, optional): Additional model parameters
        
    Returns:
        nn.Module: GNN model
    """
    if model_params is None:
        model_params = {}
    
    # Common parameters
    num_layers = model_params.get('num_layers', 2)
    dropout = model_params.get('dropout', 0.5)
    pool_method = model_params.get('pool_method', 'mean')
    use_edge_attr = model_params.get('use_edge_attr', True)
    batch_norm = model_params.get('batch_norm', True)
    
    if model_type.lower() == 'gcn':
        return GCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool_method=pool_method,
            use_edge_attr=use_edge_attr,
            batch_norm=batch_norm
        )
    
    elif model_type.lower() == 'gat':
        heads = model_params.get('heads', 4)
        concat = model_params.get('concat', True)
        
        return GAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool_method=pool_method,
            use_edge_attr=use_edge_attr,
            batch_norm=batch_norm,
            heads=heads,
            concat=concat
        )
    
    elif model_type.lower() == 'dynamic_edge':
        edge_dim = model_params.get('edge_dim', 1)
        conv_type = model_params.get('conv_type', 'gcn')
        update_edges = model_params.get('update_edges', True)
        
        return DynamicEdgeGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            pool_method=pool_method,
            use_edge_attr=use_edge_attr,
            batch_norm=batch_norm,
            edge_dim=edge_dim,
            conv_type=conv_type,
            update_edges=update_edges
        )
    
    else:
        raise ValueError(f"Unsupported GNN model type: {model_type}") 