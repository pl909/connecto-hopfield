import yaml
import logging
import random
import numpy as np
import torch
import os
import re
from typing import Dict, Any, List, Union
from sklearn.model_selection import KFold

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        dict: Validated configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {e}")
        raise

def setup_logging(log_path: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_path: Path to save log file
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Base required sections
    required_sections = ['paths', 'qc', 'data', 'training', 'analysis']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")

    # Check for *at least one* model config section
    model_section_found = False
    possible_model_sections = ['achnn_model', 'fc_mlp_model'] # Add other model types here if needed
    model_config_section = None
    for section in possible_model_sections:
        if section in config:
            model_section_found = True
            model_config_section = section
            break
    if not model_section_found:
        raise ValueError(f"Missing one of the required model sections: {possible_model_sections}")

    # Validate paths
    required_paths = [
        'base_dir',
        'phenotypic_file',
        'regional_timeseries_dir',
        'results_base_dir'
    ]
    for path in required_paths:
        if path not in config['paths']:
            raise ValueError(f"Missing required path '{path}' in config")
        if not isinstance(config['paths'][path], str):
            raise ValueError(f"Path '{path}' must be a string")
    
    # Validate data parameters
    required_data = {
        'num_regions': int,
        'tr': float,
        'seq_len': int,
        'window_step': int
    }
    for param, param_type in required_data.items():
        if param not in config['data']:
            raise ValueError(f"Missing required data parameter '{param}' in config")
        if not isinstance(config['data'][param], param_type):
            raise ValueError(f"Data parameter '{param}' must be of type {param_type}")
    
    # Validate model-specific parameters based on the found section
    if model_config_section == 'achnn_model':
        required_model = {
            'hidden_dim': int,
            'num_encoder_layers': int,
            'num_heads': int,
            'dropout': float,
            # 'update_steps_max': int, # Commenting out potentially unused params
            # 'scaling': float,
            'num_classes': int
        }
        for param, param_type in required_model.items():
            if param not in config[model_config_section]:
                raise ValueError(f"Missing required {model_config_section} parameter '{param}' in config")
            if not isinstance(config[model_config_section][param], param_type):
                raise ValueError(f"{model_config_section} parameter '{param}' must be of type {param_type}")
    elif model_config_section == 'fc_mlp_model':
        required_model = {
            'hidden_layers': list,
            'dropout_rate': float,
            'use_batch_norm': bool,
            'num_classes': int
        }
        for param, param_type in required_model.items():
            if param not in config[model_config_section]:
                raise ValueError(f"Missing required {model_config_section} parameter '{param}' in config")
            # Special check for list type
            if param == 'hidden_layers' and not isinstance(config[model_config_section][param], list):
                 raise ValueError(f"{model_config_section} parameter '{param}' must be a list")
            elif param != 'hidden_layers' and not isinstance(config[model_config_section][param], param_type):
                 raise ValueError(f"{model_config_section} parameter '{param}' must be of type {param_type}")

    # Validate training parameters
    base_required_training = {
        'seed': int,
        'device': str,
        'batch_size': int,
        'num_epochs': int,
        'learning_rate': float,
        'weight_decay': float,
        'early_stopping_patience': int,
        'gradient_clip': float,
        'mixed_precision': bool,
        'num_workers': int,
        'cv_folds': int,
        'patience': int
    }
    
    # Validate base parameters first
    for param, param_type in base_required_training.items():
        if param not in config['training']:
            raise ValueError(f"Missing required training parameter '{param}' in config")
        if not isinstance(config['training'][param], param_type):
            raise ValueError(f"Training parameter '{param}' must be of type {param_type}")

    # Validate scheduler-specific parameters
    scheduler_type = config['training'].get('scheduler_type')
    if scheduler_type == 'ReduceLROnPlateau':
        required_scheduler_params = {
            'scheduler_mode': str,
            'scheduler_metric': str,
            'scheduler_factor': float,
            'scheduler_patience': int,
            'scheduler_threshold': float,
            'scheduler_min_lr': float
        }
        for param, param_type in required_scheduler_params.items():
            if param not in config['training']:
                raise ValueError(f"Missing required ReduceLROnPlateau parameter '{param}' in config")
            if not isinstance(config['training'][param], param_type):
                raise ValueError(f"ReduceLROnPlateau parameter '{param}' must be of type {param_type}")
    elif scheduler_type == 'OneCycleLR':
         required_scheduler_params = {
            'max_lr': float,
            # Add other OneCycleLR specific params if needed, like pct_start, div_factor etc.
            # For now, just checking max_lr as it's the most critical one defined.
         }
         for param, param_type in required_scheduler_params.items():
            if param not in config['training']:
                raise ValueError(f"Missing required OneCycleLR parameter '{param}' in config")
            if not isinstance(config['training'][param], param_type):
                 raise ValueError(f"OneCycleLR parameter '{param}' must be of type {param_type}")
    elif scheduler_type is not None:
        logging.warning(f"Unknown scheduler_type '{scheduler_type}'. Skipping scheduler-specific validation.")

    # Validate device
    if config['training']['device'] not in ['cuda', 'cpu']:
        raise ValueError("Device must be either 'cuda' or 'cpu'")
    
    # Validate batch size
    if config['training']['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    if not (config['training']['batch_size'] & (config['training']['batch_size'] - 1) == 0):
        logging.warning("batch_size is not a power of 2, which might impact performance")
    
    # Log validation success
    logging.info("Configuration validation successful")

def get_subject_id_from_path(file_path: str) -> str:
    """
    Extract subject ID from file path.
    
    Args:
        file_path: Path to subject file
        
    Returns:
        str: Subject ID
    """
    return os.path.basename(file_path).split('_')[0]

def create_fold_indices(clinical_data, n_folds, subject_id_col):
    """
    Create fold indices for cross-validation.
    
    Args:
        clinical_data: DataFrame containing clinical data
        n_folds: Number of folds for cross-validation
        subject_id_col: Name of the column containing subject IDs
        
    Returns:
        List of dictionaries, each containing train, val, and test indices for a fold
    """
    # Extract unique subject IDs
    subjects = clinical_data[subject_id_col].unique()
    n_subjects = len(subjects)
    
    # Create KFold cross-validator
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize fold indices
    fold_indices = []
    
    # Create indices for each fold
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(subjects)):
        # Split train_val into train and val
        n_val = len(train_val_idx) // 5  # Use 20% of train for validation
        train_idx = train_val_idx[:-n_val]
        val_idx = train_val_idx[-n_val:]
        
        # Get subject IDs for each set
        train_subjects = subjects[train_idx]
        val_subjects = subjects[val_idx]
        test_subjects = subjects[test_idx]
        
        # Get indices in clinical_data for each set
        train_indices = clinical_data[clinical_data[subject_id_col].isin(train_subjects)].index.tolist()
        val_indices = clinical_data[clinical_data[subject_id_col].isin(val_subjects)].index.tolist()
        test_indices = clinical_data[clinical_data[subject_id_col].isin(test_subjects)].index.tolist()
        
        # Store indices for this fold
        fold_indices.append({
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        })
        
        # Log fold distribution
        logging.info(f"Fold {fold}: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects, {len(test_subjects)} test subjects")
    
    return fold_indices 