#!/usr/bin/env python
"""
Simplified regression training script for a small number of subjects.
This loads data for a few sample subjects and runs a full training cycle.
"""
import os
import sys
import time
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime
from sklearn.model_selection import KFold

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.data_loader import load_timeseries_data, create_data_loaders
from src.training import train_model_regression
from src.utils import set_seed, create_fold_indices, setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Run a minimal regression training pipeline with a few subjects"""
    logging.info("Loading config...")
    config_path = 'configs/achnn_regression_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config['experiment']['results_path'], f"minimal_test_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(results_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Test with a few known subject IDs (these are known to work)
    test_subject_ids = ['50003', '50004', '50005', '50006', '50007']
    logging.info(f"Training with subjects: {test_subject_ids}")
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Set device
    device_name = config['training']['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU instead")
        device_name = 'cpu'
    device = torch.device(device_name)
    
    try:
        # Load the data for these subjects
        start_time = time.time()
        subject_data_dict, clinical_data = load_timeseries_data(
            config, subject_ids=test_subject_ids, task_type='regression'
        )
        
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        logging.info(f"Successfully loaded data for {len(subject_data_dict)} subjects")
        
        # Print info about each subject's data
        for subject_id, windows in subject_data_dict.items():
            logging.info(f"Subject {subject_id}: {len(windows)} windows, shape: {windows.shape}")
            
            # Show ADOS_TOTAL score for this subject if available
            regression_target = config.get('data', {}).get('regression_target', 'ADOS_TOTAL')
            subject_id_col = config.get('data', {}).get('subject_id_col', 'SUB_ID')
            
            if regression_target in clinical_data.columns:
                subject_rows = clinical_data[clinical_data[subject_id_col] == subject_id]
                if len(subject_rows) > 0:
                    target_value = subject_rows[regression_target].iloc[0]
                    logging.info(f"Subject {subject_id} {regression_target}: {target_value}")
        
        # Do a simple 2-fold cross-validation
        n_folds = 2
        config['training']['num_epochs'] = 15  # Reduce epochs for test
        
        # Create fold indices
        fold_indices = create_fold_indices(clinical_data, n_folds, subject_id_col)
        
        # Initialize results list
        fold_results = []
        
        # Train for each fold
        for fold_idx in range(n_folds):
            logging.info(f"===== Starting Fold {fold_idx+1}/{n_folds} =====")
            
            # Create fold directory
            fold_dir = os.path.join(results_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Create data loaders for this fold
            train_loader, val_loader, test_loader = create_data_loaders(
                subject_data_dict, 
                clinical_data,
                config,
                fold_indices[fold_idx],
                task_type='regression'
            )
            
            # Train model
            test_results = train_model_regression(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                fold_dir=fold_dir,
                device=device,
                fold_idx=fold_idx
            )
            
            # Store results
            fold_results.append({
                'fold': fold_idx,
                'test_mse': test_results['test_mse'],
                'test_mae': test_results['test_mae'],
                'test_r2': test_results.get('test_r2', float('nan'))
            })
        
        # Calculate average results
        avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
        avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
        avg_test_r2 = np.mean([r['test_r2'] for r in fold_results if not np.isnan(r['test_r2'])])
        
        # Log summary results
        logging.info("===== Cross-Validation Results =====")
        logging.info(f"Average Test MSE: {avg_test_mse:.4f}")
        logging.info(f"Average Test MAE: {avg_test_mae:.4f}")
        logging.info(f"Average Test R²: {avg_test_r2:.4f}")
        
        # Save summary results
        summary_results = {
            'avg_test_mse': float(avg_test_mse),
            'avg_test_mae': float(avg_test_mae),
            'avg_test_r2': float(avg_test_r2),
            'fold_results': fold_results
        }
        
        with open(os.path.join(results_dir, 'summary_results.json'), 'w') as f:
            json.dump(summary_results, f, indent=4)
        
        logging.info(f"Training completed. Results saved to {results_dir}")
            
    except Exception as e:
        logging.error(f"❌ Error in training: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 