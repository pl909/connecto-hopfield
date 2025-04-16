import numpy as np
import torch
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
import logging
import scipy.sparse as sp
import os
from torch.utils.data import Dataset, Subset, random_split
from torch_geometric.data import Data, Dataset as PyGDataset, InMemoryDataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch_geometric.data as geom_data
from torch_geometric.utils import dense_to_sparse, to_undirected
import pickle
import networkx as nx
from tqdm import tqdm
from torch_geometric.transforms import BaseTransform

# Setup logger
logger = logging.getLogger(__name__)

# Check if torch_geometric is installed, provide helpful error if not
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
except ImportError:
    error_msg = """
    PyTorch Geometric is required for GNN utilities but not installed.
    Please install it with:
    
    pip install torch-geometric
    
    For CUDA support, follow installation instructions at:
    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    """
    logging.error(error_msg)
    raise ImportError(error_msg)


def fc_matrix_to_graph(
    fc_matrix: np.ndarray,
    labels: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    self_loops: bool = False,
    sparse_format: bool = True
) -> Data:
    """
    Convert a functional connectivity matrix to a PyTorch Geometric graph.
    
    Args:
        fc_matrix: An NxN functional connectivity matrix
        labels: Optional class labels for the graph (for classification tasks)
        threshold: Optional threshold to apply to connectivity (remove weak connections)
        self_loops: Whether to include self-loops in the graph
        sparse_format: Whether to use sparse format for edge indices
        
    Returns:
        torch_geometric.data.Data: Graph data object with connectivity information
    """
    # Create node features (use 1D feature per node initially)
    num_regions = fc_matrix.shape[0]
    x = torch.ones(num_regions, 1, dtype=torch.float)
    
    # Apply threshold if specified (remove weak connections)
    if threshold is not None:
        fc_matrix = fc_matrix.copy()
        fc_matrix[np.abs(fc_matrix) < threshold] = 0
    
    # Remove self loops if not desired
    if not self_loops:
        np.fill_diagonal(fc_matrix, 0)
    
    # Convert to edge_index format (COO format for PyTorch Geometric)
    if sparse_format:
        # Convert to scipy sparse matrix first
        sparse_fc = sp.coo_matrix(fc_matrix)
        edge_index = torch.tensor(np.vstack((sparse_fc.row, sparse_fc.col)), dtype=torch.long)
        edge_attr = torch.tensor(sparse_fc.data, dtype=torch.float).reshape(-1, 1)
    else:
        # Dense format (all non-zero edges)
        rows, cols = np.where(fc_matrix != 0)
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_attr = torch.tensor([fc_matrix[i, j] for i, j in zip(rows, cols)], dtype=torch.float).reshape(-1, 1)
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )
    
    return graph_data


def batch_fc_matrices_to_graphs(
    fc_matrices: np.ndarray,
    labels: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    self_loops: bool = False
) -> Batch:
    """
    Convert a batch of functional connectivity matrices to PyTorch Geometric graphs.
    
    Args:
        fc_matrices: Batch of NxN functional connectivity matrices [batch_size, N, N]
        labels: Optional class labels for each graph [batch_size]
        threshold: Optional threshold to apply to connectivity
        self_loops: Whether to include self-loops in the graphs
        
    Returns:
        torch_geometric.data.Batch: Batched graph data object
    """
    graph_list = []
    
    for i, fc_matrix in enumerate(fc_matrices):
        label = labels[i].unsqueeze(0) if labels is not None else None
        graph = fc_matrix_to_graph(
            fc_matrix=fc_matrix,
            labels=label,
            threshold=threshold,
            self_loops=self_loops
        )
        graph_list.append(graph)
    
    return Batch.from_data_list(graph_list)


class FCGraphDataset(Dataset):
    """Dataset for converting functional connectivity matrices to graph data objects.
    
    This dataset takes functional connectivity (FC) matrices and converts them into
    graph structures suitable for use with Graph Neural Networks (GNNs). 
    Each FC matrix is treated as an adjacency matrix of a graph.
    """
    
    def __init__(self, fc_matrices: np.ndarray, labels: np.ndarray,
                 node_features: Optional[np.ndarray] = None,
                 region_coords: Optional[np.ndarray] = None,
                 threshold: Optional[float] = None,
                 self_loops: bool = False,
                 normalize_edges: bool = True,
                 binary_edges: bool = False,
                 undirected: bool = True,
                 transform: Optional[Callable] = None):
        """Initialize the dataset.
        
        Args:
            fc_matrices (np.ndarray): Functional connectivity matrices [n_samples, n_regions, n_regions]
            labels (np.ndarray): Labels for each sample [n_samples]
            node_features (np.ndarray, optional): Features for each node/region [n_samples, n_regions, n_features]
                If None, the FC row-wise average is used as node features
            region_coords (np.ndarray, optional): 3D coordinates for each region [n_regions, 3]
            threshold (float, optional): Threshold to filter weak connections
            self_loops (bool): Whether to include self-loops in the graph
            normalize_edges (bool): Whether to normalize edge weights to [0, 1]
            binary_edges (bool): If True, edges are binary (1 if value > threshold, else 0)
            undirected (bool): If True, ensures the graph is undirected
            transform (callable, optional): Transform to apply to each sample
        """
        super(FCGraphDataset, self).__init__(transform=transform)
        
        # Store dataset parameters
        self.fc_matrices = fc_matrices
        self.labels = labels
        self.node_features = node_features
        self.region_coords = region_coords
        self.threshold = threshold
        self.self_loops = self_loops
        self.normalize_edges = normalize_edges
        self.binary_edges = binary_edges
        self.undirected = undirected
        
        # Verify inputs
        self._verify_inputs()
        
        # Store dataset size
        self.n_samples = self.fc_matrices.shape[0]
        self.n_regions = self.fc_matrices.shape[1]
        
        # Log dataset info
        logger.info(f"Created FCGraphDataset with {self.n_samples} samples and {self.n_regions} regions")
        
    def _verify_inputs(self):
        """Verify that the input shapes are compatible."""
        if len(self.fc_matrices.shape) != 3:
            raise ValueError(f"FC matrices should be 3D: [n_samples, n_regions, n_regions], "
                             f"got {self.fc_matrices.shape}")
        
        if self.fc_matrices.shape[1] != self.fc_matrices.shape[2]:
            raise ValueError(f"FC matrices should be square, but got shape "
                             f"{self.fc_matrices.shape[1]} x {self.fc_matrices.shape[2]}")
        
        if len(self.labels.shape) != 1 or self.labels.shape[0] != self.fc_matrices.shape[0]:
            raise ValueError(f"Labels should be 1D with shape [n_samples], "
                             f"got {self.labels.shape}")
        
        if self.node_features is not None:
            if len(self.node_features.shape) != 3:
                raise ValueError(f"Node features should be 3D: [n_samples, n_regions, n_features], "
                                f"got {self.node_features.shape}")
            
            if self.node_features.shape[0] != self.fc_matrices.shape[0]:
                raise ValueError(f"Node features should have same number of samples as FC matrices, "
                                f"got {self.node_features.shape[0]} vs. {self.fc_matrices.shape[0]}")
            
            if self.node_features.shape[1] != self.fc_matrices.shape[1]:
                raise ValueError(f"Node features should have same number of regions as FC matrices, "
                                f"got {self.node_features.shape[1]} vs. {self.fc_matrices.shape[1]}")
        
        if self.region_coords is not None:
            if len(self.region_coords.shape) != 2 or self.region_coords.shape[0] != self.fc_matrices.shape[1]:
                raise ValueError(f"Region coordinates should be 2D: [n_regions, 3], "
                                f"got {self.region_coords.shape}")
    
    def len(self):
        """Return the number of samples in the dataset."""
        return self.n_samples
    
    def get(self, idx):
        """Get the graph data object for a specific sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Data: PyTorch Geometric Data object containing the graph
        """
        # Get the FC matrix for this sample
        fc_matrix = self.fc_matrices[idx].copy()
        label = self.labels[idx]
        
        # Convert FC matrix to graph
        graph = self._fc_to_graph(fc_matrix, idx)
        
        # Add label
        graph.y = torch.tensor(label, dtype=torch.long)
        
        return graph
    
    def _fc_to_graph(self, fc_matrix, idx):
        """Convert a functional connectivity matrix to a graph.
        
        Args:
            fc_matrix (np.ndarray): Functional connectivity matrix [n_regions, n_regions]
            idx (int): Index of the sample
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        # Apply threshold if specified
        if self.threshold is not None:
            mask = np.abs(fc_matrix) > self.threshold
            fc_matrix = fc_matrix * mask
        
        # Remove self-loops if specified
        if not self.self_loops:
            np.fill_diagonal(fc_matrix, 0)
        
        # Handle edge weights
        if self.binary_edges:
            # Convert to binary edges
            edge_weights = (np.abs(fc_matrix) > 0).astype(np.float32)
        else:
            # Use continuous edge weights
            edge_weights = fc_matrix.copy()
            
            # Normalize edge weights if specified
            if self.normalize_edges:
                # Scale to [0, 1] range
                min_val = np.min(edge_weights)
                max_val = np.max(edge_weights)
                if max_val > min_val:
                    edge_weights = (edge_weights - min_val) / (max_val - min_val)
        
        # Convert to sparse format
        edge_index, edge_attr = dense_to_sparse(torch.tensor(edge_weights, dtype=torch.float))
        
        # Make undirected if specified
        if self.undirected:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        
        # Prepare node features
        if self.node_features is not None:
            # Use provided node features
            x = torch.tensor(self.node_features[idx], dtype=torch.float)
        else:
            # Use row-wise average of FC matrix as node features
            # This represents the average connectivity of each region
            node_feats = np.mean(np.abs(fc_matrix), axis=1, keepdims=True)
            x = torch.tensor(node_feats, dtype=torch.float)
        
        # Create the graph data object
        graph_data = Data(
            x=x,                           # Node features
            edge_index=edge_index,         # Edge indices
            edge_attr=edge_attr.unsqueeze(-1),  # Edge attributes
        )
        
        # Add node positions if available
        if self.region_coords is not None:
            graph_data.pos = torch.tensor(self.region_coords, dtype=torch.float)
        
        return graph_data


class BrainConnectivityDataset(InMemoryDataset):
    """PyTorch Geometric dataset for brain connectivity data.
    
    This dataset processes functional connectivity matrices into graph data objects
    for use with Graph Neural Networks.
    """
    
    def __init__(self, root: str, threshold: float = 0.0, 
                 node_feature_type: str = 'degree',
                 normalize_method: str = 'none', 
                 transform=None, pre_transform=None):
        """Initialize the brain connectivity dataset.
        
        Args:
            root (str): Root directory for the dataset
            threshold (float): Threshold for filtering edges
            node_feature_type (str): Type of node features to compute
                ('degree', 'eigenvector', 'betweenness', 'identity', 'onehot', 'strength')
            normalize_method (str): Method for normalizing edge weights
                ('none', 'abs', 'scale', 'negative_shift', 'positive')
            transform (callable, optional): Transform to apply to each data object
            pre_transform (callable, optional): Transform to apply to each data object before saving
        """
        # Store processing parameters
        self.threshold = threshold
        self.node_feature_type = node_feature_type
        self.normalize_method = normalize_method
        
        super(BrainConnectivityDataset, self).__init__(root, transform, pre_transform)
        # Load the processed data once it's generated
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return the names of the raw files."""
        # This is the file we expect to exist in the raw_dir
        return ['raw_data.pt']
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return the names of the processed files."""
        # Include normalization method in the filename for clarity
        return [f'brain_connectivity_data_{self.node_feature_type}_norm-{self.normalize_method}_t{self.threshold}.pt']
    
    def download(self):
        """Download data if raw files don't exist.
        
        In this setup, the raw data is expected to be pre-saved by the DataLoader, 
        so this method does nothing.
        """
        # Raw file should be saved by the GraphDataLoader before dataset initialization
        logger.info(f"BrainConnectivityDataset expects raw file {self.raw_paths[0]} to exist. Skipping download.")
        pass
        # No need to save here anymore
        # if not os.path.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # torch.save((self.raw_matrices, self.labels), os.path.join(self.raw_dir, 'raw_data.pt'), pickle_protocol=5)
        # logger.info(f"Saved raw data to {os.path.join(self.raw_dir, 'raw_data.pt')}")
    
    def process(self):
        """Process raw data into graph data objects."""
        # Create the processed directory if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        # Load raw data - expects raw_paths[0] to exist
        raw_data_path = self.raw_paths[0]
        if not os.path.exists(raw_data_path):
             # If download() didn't create it (which it won't now), raise an error
             raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. " 
                                   f"Ensure it's saved before initializing BrainConnectivityDataset.")

        logger.info(f"Loading raw data from {raw_data_path} for processing...")
        # Explicitly set weights_only=False as we are loading numpy arrays, not just model weights.
        # We trust the source as we just saved this file.
        matrices, labels = torch.load(raw_data_path, weights_only=False) 

        # Process each connectivity matrix into a graph
        data_list = []
        
        logger.info(f"Processing {len(matrices)} connectivity matrices into graphs")
        for i in tqdm(range(len(matrices))):
            data = functional_connectivity_to_graph(
                matrices[i], 
                threshold=self.threshold,
                node_feature_type=self.node_feature_type,
                normalize_method=self.normalize_method
            )
            data.y = torch.tensor([labels[i]], dtype=torch.long)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved {len(data_list)} processed graphs to {self.processed_paths[0]}")


def functional_connectivity_to_graph(
    connectivity_matrix: np.ndarray,
    threshold: float = 0.0,
    node_feature_type: str = 'degree',
    normalize_method: str = 'none'
) -> Data:
    """Convert a functional connectivity matrix to a PyTorch Geometric Data object.
    
    Args:
        connectivity_matrix (np.ndarray): Functional connectivity matrix
        threshold (float): Threshold for filtering edges
        node_feature_type (str): Type of node features to compute
        normalize_method (str): Method for normalizing edge weights
    
    Returns:
        Data: PyTorch Geometric data object
    """
    # Convert to numpy array if needed
    if isinstance(connectivity_matrix, torch.Tensor):
        connectivity_matrix = connectivity_matrix.numpy()
    
    # Normalize edge weights
    normalized_matrix = normalize_edge_weights(connectivity_matrix, method=normalize_method)
    
    # Apply threshold to create an adjacency matrix
    if threshold > 0:
        # Keep only edges above threshold
        thresholded_matrix = np.copy(normalized_matrix)
        thresholded_matrix[np.abs(thresholded_matrix) < threshold] = 0
    else:
        thresholded_matrix = normalized_matrix
    
    # Create edge indices and attributes
    edge_index = []
    edge_attr = []
    
    for i in range(thresholded_matrix.shape[0]):
        for j in range(thresholded_matrix.shape[1]):
            if thresholded_matrix[i, j] != 0:
                edge_index.append([i, j])
                edge_attr.append(thresholded_matrix[i, j])
    
    if len(edge_index) == 0:
        # If no edges remain after thresholding, create a self-loop for each node
        logger.warning(f"No edges remain after thresholding with value {threshold}. Creating self-loops.")
        edge_index = [[i, i] for i in range(thresholded_matrix.shape[0])]
        edge_attr = [1.0] * len(edge_index)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    
    # Compute node features
    x = compute_node_features(thresholded_matrix, feature_type=node_feature_type)
    
    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def compute_node_features(
    adjacency_matrix: np.ndarray,
    feature_type: str = 'degree'
) -> torch.Tensor:
    """Compute node features based on the adjacency matrix.
    
    Args:
        adjacency_matrix (np.ndarray): Adjacency matrix
        feature_type (str): Type of node features to compute
            ('degree', 'eigenvector', 'betweenness', 'identity', 'onehot', 'strength')
    
    Returns:
        torch.Tensor: Node features tensor
    """
    num_nodes = adjacency_matrix.shape[0]
    
    if feature_type == 'identity':
        # Identity features (no node features)
        features = np.eye(num_nodes)
    
    elif feature_type == 'onehot':
        # One-hot encoding
        features = np.eye(num_nodes)
    
    elif feature_type == 'degree':
        # Degree features
        out_degree = np.sum(np.abs(adjacency_matrix) > 0, axis=1).reshape(-1, 1)
        in_degree = np.sum(np.abs(adjacency_matrix) > 0, axis=0).reshape(-1, 1)
        features = np.hstack([out_degree, in_degree])
    
    elif feature_type == 'strength':
        # Connection strength features
        out_strength = np.sum(adjacency_matrix, axis=1).reshape(-1, 1)
        in_strength = np.sum(adjacency_matrix, axis=0).reshape(-1, 1)
        features = np.hstack([out_strength, in_strength])
    
    elif feature_type in ['eigenvector', 'betweenness', 'closeness', 'pagerank']:
        # Create a NetworkX graph for centrality calculations
        G = nx.from_numpy_array(np.abs(adjacency_matrix), create_using=nx.DiGraph())
        
        if feature_type == 'eigenvector':
            # Eigenvector centrality
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=1000)
                features = np.array([[centrality[i]] for i in range(num_nodes)])
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                logger.warning("Eigenvector centrality calculation failed. Using degree centrality instead.")
                centrality = nx.degree_centrality(G)
                features = np.array([[centrality[i]] for i in range(num_nodes)])
        
        elif feature_type == 'betweenness':
            # Betweenness centrality
            centrality = nx.betweenness_centrality(G)
            features = np.array([[centrality[i]] for i in range(num_nodes)])
        
        elif feature_type == 'closeness':
            # Closeness centrality
            centrality = nx.closeness_centrality(G)
            features = np.array([[centrality[i]] for i in range(num_nodes)])
        
        elif feature_type == 'pagerank':
            # PageRank centrality
            centrality = nx.pagerank(G)
            features = np.array([[centrality[i]] for i in range(num_nodes)])
    
    else:
        # Default to ones if feature type is not recognized
        logger.warning(f"Unknown node feature type: {feature_type}. Using ones.")
        features = np.ones((num_nodes, 1))
    
    return torch.tensor(features, dtype=torch.float)


def normalize_edge_weights(
    connectivity_matrix: np.ndarray,
    method: str = 'none'
) -> np.ndarray:
    """Normalize edge weights in the connectivity matrix.
    
    Args:
        connectivity_matrix (np.ndarray): Connectivity matrix
        method (str): Normalization method
            ('none', 'abs', 'scale', 'negative_shift', 'positive')
    
    Returns:
        np.ndarray: Normalized connectivity matrix
    """
    if method == 'none':
        # No normalization
        return connectivity_matrix
    
    elif method == 'abs':
        # Take absolute value
        return np.abs(connectivity_matrix)
    
    elif method == 'scale':
        # Scale to [0, 1]
        min_val = np.min(connectivity_matrix)
        max_val = np.max(connectivity_matrix)
        if min_val == max_val:
            return np.zeros_like(connectivity_matrix)
        return (connectivity_matrix - min_val) / (max_val - min_val)
    
    elif method == 'negative_shift':
        # Shift negative values to 0
        normalized = np.copy(connectivity_matrix)
        normalized[normalized < 0] = 0
        return normalized
    
    elif method == 'positive':
        # Keep only positive connections
        normalized = np.copy(connectivity_matrix)
        normalized[normalized <= 0] = 0
        return normalized
    
    elif method == 'symmetric_abs':
        # Ensure symmetry and take absolute values
        return np.abs((connectivity_matrix + connectivity_matrix.T) / 2)
    
    elif method == 'znorm':
        # Z-score normalization
        mean = np.mean(connectivity_matrix)
        std = np.std(connectivity_matrix)
        if std == 0:
            return np.zeros_like(connectivity_matrix)
        return (connectivity_matrix - mean) / std
    
    else:
        logger.warning(f"Unknown normalization method: {method}. No normalization applied.")
        return connectivity_matrix


class GraphDataLoader:
    """Data loader for brain connectivity graphs.
    
    This class handles loading and preprocessing brain connectivity data
    for training GNN models.
    """
    
    def __init__(self, data_path: str, 
                 batch_size: int = 32,
                 threshold: float = 0.0,
                 node_feature_type: str = 'degree',
                 normalize_method: str = 'none',
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 num_workers: int = 4):
        """Initialize the data loader.
        
        Args:
            data_path (str): Path to the raw data
            batch_size (int): Batch size for DataLoader
            threshold (float): Threshold for filtering edges
            node_feature_type (str): Type of node features to compute
            normalize_method (str): Method for normalizing edge weights
            test_size (float): Proportion of data to use for testing
            val_size (float): Proportion of data to use for validation
            random_state (int): Random seed for data splitting
            num_workers (int): Number of worker processes for data loading. Default: 4.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.threshold = threshold
        self.node_feature_type = node_feature_type
        self.normalize_method = normalize_method
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def load_data(self, matrices: np.ndarray, labels: np.ndarray):
        """Load and preprocess data.
        
        Args:
            matrices (np.ndarray): Array of connectivity matrices (already filtered)
            labels (np.ndarray): Array of labels corresponding to each matrix (already filtered)
        """
        # Define the root directory for the dataset
        root = os.path.join(self.data_path, 'brain_connectivity_dataset')
        raw_dir = os.path.join(root, 'raw')
        processed_dir = os.path.join(root, 'processed')

        # Ensure raw and processed directories exist
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Define the path for the raw data file
        raw_file_path = os.path.join(raw_dir, 'raw_data.pt')

        # Save the filtered matrices and labels to the raw directory *before* creating the dataset
        logger.info(f"Saving filtered raw data ({len(matrices)} samples) to {raw_file_path}...")
        torch.save((matrices, labels), raw_file_path, pickle_protocol=5)
        logger.info("Raw data saved successfully.")

        # Now create the dataset. It will find the raw data in raw_paths[0]
        # and process it if the processed file doesn't exist.
        logger.info(f"Initializing BrainConnectivityDataset from root: {root}")
        dataset = BrainConnectivityDataset(
            root=root,
            threshold=self.threshold,
            node_feature_type=self.node_feature_type,
            normalize_method=self.normalize_method
        )
        logger.info(f"BrainConnectivityDataset initialized. Found {len(dataset)} samples.")

        # Split the dataset into train, validation, and test sets
        num_samples = len(dataset)
        indices = list(range(num_samples))
        
        # Stratify based on the labels loaded from the dataset, not the original labels
        # This ensures stratification matches the actual dataset after potential processing issues
        dataset_labels = np.array([data.y.item() for data in dataset])
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state, stratify=dataset_labels
        )
        
        # Second split: separate validation set from training set
        # Ensure we don't divide by zero if test_size is 1.0
        if self.test_size < 1.0:
             val_size_adjusted = self.val_size / (1 - self.test_size)
             train_indices, val_indices = train_test_split(
                 train_val_indices, test_size=val_size_adjusted, 
                 random_state=self.random_state, stratify=dataset_labels[train_val_indices]
             )
        else: # Handle edge case where test_size = 1.0 (no training/validation needed)
             train_indices = []
             val_indices = []
             logger.warning("test_size is 1.0, resulting in empty train/validation sets.")

        # Create data loaders with num_workers
        logger.info(f"Creating DataLoaders with batch_size={self.batch_size} and num_workers={self.num_workers}")
        self.train_loader = DataLoader(
            dataset=[dataset[i] for i in train_indices],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            dataset=[dataset[i] for i in val_indices],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            dataset=[dataset[i] for i in test_indices],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Dataset split: {len(train_indices)} training, "
                    f"{len(val_indices)} validation, {len(test_indices)} test samples")
        logger.info(f"DataLoaders created successfully.")
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get the data loaders.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.train_loader, self.val_loader, self.test_loader


def visualize_brain_graph(graph_data, title="Brain Connectivity Graph", 
                          node_size=30, edge_width_scale=5.0, 
                          colormap='viridis', edge_threshold=None,
                          show_negative=True, show_positive=True):
    """Visualize a brain connectivity graph in 3D.
    
    Args:
        graph_data (torch_geometric.data.Data): Graph data object
        title (str): Title for the visualization
        node_size (float): Size of nodes in the visualization
        edge_width_scale (float): Scale factor for edge widths
        colormap (str): Colormap for edge colors
        edge_threshold (float, optional): Threshold to filter edges by absolute weight
        show_negative (bool): Whether to show negative edges
        show_positive (bool): Whether to show positive edges
        
    Returns:
        matplotlib.figure.Figure: 3D visualization of the brain graph
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logging.error("matplotlib is required for visualization")
        return None
    
    # Check if coordinates are available
    if not hasattr(graph_data, 'pos'):
        logging.error("No node coordinates (pos) found in graph data")
        return None
    
    # Get node positions, edge indices, and edge attributes
    pos = graph_data.pos.numpy()
    edge_index = graph_data.edge_index.numpy()
    edge_attr = graph_data.edge_attr.numpy().flatten()
    
    # Apply threshold if specified
    if edge_threshold is not None:
        mask = np.abs(edge_attr) > edge_threshold
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
    
    # Filter edges based on sign if specified
    if not show_negative:
        pos_mask = edge_attr > 0
        edge_index = edge_index[:, pos_mask]
        edge_attr = edge_attr[pos_mask]
    elif not show_positive:
        neg_mask = edge_attr < 0
        edge_index = edge_index[:, neg_mask]
        edge_attr = edge_attr[neg_mask]
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot nodes
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=node_size, color='gray')
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        x = np.array([pos[src, 0], pos[dst, 0]])
        y = np.array([pos[src, 1], pos[dst, 1]])
        z = np.array([pos[src, 2], pos[dst, 2]])
        
        # Set edge color and width based on weight
        weight = edge_attr[i]
        width = np.abs(weight) * edge_width_scale
        color = 'red' if weight < 0 else 'blue'
        
        ax.plot(x, y, z, linewidth=width, color=color, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Positive connection'),
        Line2D([0], [0], color='red', lw=2, label='Negative connection'),
        Line2D([0], [0], marker='o', color='gray', label='Brain region',
               markersize=5, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig 