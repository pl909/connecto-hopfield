import yaml
import logging
import random
import numpy as np
import torch
import os
import re

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        raise

def setup_logging(log_path, level=logging.INFO):
    """Sets up logging to file and console."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

def get_subject_id_from_path(file_path):
    """Extracts subject ID (e.g., 'sub-XX') from a file path."""
    match = re.search(r'sub-(\d+)', file_path)
    if match:
        return f"sub-{match.group(1)}"
    else:
        logging.warning(f"Could not extract subject ID from path: {file_path}")
        return None 