import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_graph(node_features, k, device):
    """Builds a k-NN graph using scikit-learn for efficiency."""
    num_nodes = node_features.shape[0]
    if num_nodes <= k:
        print(f"Warning: Number of nodes ({num_nodes}) is less than or equal to k ({k}). Adjusting k to {max(1, num_nodes - 1)}.")
        k = max(1, num_nodes - 1)
        if k == 0:
             print("Error: Cannot build graph with only one node.")
             return torch.empty((2, 0), dtype=torch.long, device=device)

    print(f"Building k-NN graph (k={k})...")
    features_np = node_features.detach().cpu().numpy()
    if not np.all(np.isfinite(features_np)):
        print("Warning: Non-finite values detected in node features before k-NN. Replacing with zeros.")
        features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='cosine', n_jobs=-1)
        nn_model.fit(features_np)
        distances, indices = nn_model.kneighbors(features_np)
    except ValueError as e:
        print(f"Error during k-NN fitting (possibly due to feature values): {e}. Trying Euclidean.")
        try:
            nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean', n_jobs=-1)
            nn_model.fit(features_np)
            distances, indices = nn_model.kneighbors(features_np)
        except Exception as e2:
            print(f"Error during k-NN fallback: {e2}. Returning empty graph.")
            return torch.empty((2, 0), dtype=torch.long, device=device)

    row_indices = []
    col_indices = []
    for i in range(num_nodes):
        for j_idx in range(1, k + 1):
             if j_idx < indices.shape[1]:
                neighbor_node_idx = indices[i, j_idx]
                if 0 <= neighbor_node_idx < num_nodes:
                    row_indices.append(i)
                    col_indices.append(neighbor_node_idx)
                else:
                    print(f"Warning: Invalid neighbor index {neighbor_node_idx} found for node {i}.")
             else:
                  print(f"Warning: Neighbor index {j_idx} out of bounds for node {i}.")

    if not row_indices:
         print("Warning: No edges created in k-NN graph. Check node features and k.")
         return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long).contiguous().to(device)
    print(f"Graph built with {num_nodes} nodes and {edge_index.shape[1]} edges.")
    return edge_index