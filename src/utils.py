import random
import numpy as np
import torch

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")

def get_device(config_device='cuda'):
    """Gets the appropriate torch device."""
    if config_device == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if config_device == 'cuda':
             print("CUDA specified but not available. Using CPU.")
        device = torch.device("cpu")
        print("Using CPU")
    return device