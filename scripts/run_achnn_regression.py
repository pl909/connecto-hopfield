#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import yaml
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold
import json
import argparse

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data_loader import load_timeseries_data, create_data_loaders
from training import train_model_regression
from utils import set_seed, create_fold_indices, setup_logging

def main():
    """Main function to run the regression pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ACHNN regression training')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited subjects')
    parser.add_argument('--max_subjects', type=int, default=10, help='Maximum number of subjects to use in test mode')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config file')
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'configs', 'achnn_regression_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(config['experiment']['results_path'], f"{config['experiment']['name']}_{timestamp}")
    
    # Add test indicator to results directory if in test mode
    if args.test:
        results_dir = f"{results_dir}_test"
        
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(results_dir, 'training.log')
    setup_logging(log_file)
    
    # Log configuration and mode
    logging.info(f"Starting ACHNN regression training with config from: {config_path}")
    if args.test:
        logging.info(f"Running in TEST MODE with maximum {args.max_subjects} subjects")
    logging.info(f"Results will be saved to: {results_dir}")
    
    # Save configuration to results directory
    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set random seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Set device
    device_name = config['training']['device']
    if device_name == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU instead")
        device_name = 'cpu'
    device = torch.device(device_name)
    
    # Load data
    logging.info("Loading timeseries data...")
    start_time = time.time()
    
    # In test mode, select specific subjects if desired
    test_subject_ids = None
    if args.test:
        # Get subject IDs from successful test (if available)
        try:
            clinical_data_path = config['data']['clinical_data_path']
            clinical_df = pd.read_csv(clinical_data_path)
            
            # Select subjects that have valid ADOS_TOTAL scores
            target_column = config['data']['regression_target']
            valid_subjects = clinical_df[
                ~clinical_df[target_column].isna() &
                (clinical_df[target_column] > 0) &
                (clinical_df['FILE_ID'] != 'no_filename')
            ]
            
            # Get a limited number of subjects
            if len(valid_subjects) > args.max_subjects:
                valid_subjects = valid_subjects.iloc[:args.max_subjects]
                
            # Extract subject IDs - try SUB_ID or subject column
            subject_id_col = config['data'].get('subject_id_col', 'SUB_ID')
            if subject_id_col in valid_subjects.columns:
                test_subject_ids = valid_subjects[subject_id_col].astype(str).tolist()
                logging.info(f"Selected {len(test_subject_ids)} test subjects with valid targets")
            
        except Exception as e:
            logging.warning(f"Could not pre-select test subjects: {e}")
            
    # Update: Enhanced data loading with improved subject ID handling
    try:
        subject_data_dict, clinical_data = load_timeseries_data(
            config, 
            subject_ids=test_subject_ids,
            task_type='regression'
        )
        
        if len(subject_data_dict) == 0:
            logging.warning("No subjects were loaded. This may indicate issues with file paths or subject IDs.")
            logging.warning("Checking clinical data columns for debugging...")
            
            # Load clinical data directly for debugging
            clinical_data_path = config['data']['clinical_data_path']
            if os.path.exists(clinical_data_path):
                clinical_df = pd.read_csv(clinical_data_path)
                logging.info(f"Clinical data loaded from {clinical_data_path}, shape: {clinical_df.shape}")
                
                # Check for potential subject ID columns
                id_columns = [col for col in clinical_df.columns if 'ID' in col or 'id' in col or 'subject' in col.lower()]
                logging.info(f"Potential subject ID columns: {id_columns}")
                
                # Check file_id_col if specified
                file_id_col = config['data'].get('file_id_col')
                if file_id_col and file_id_col in clinical_df.columns:
                    logging.info(f"Sample {file_id_col} values: {clinical_df[file_id_col].head(5).tolist()}")
            
            # Check timeseries directory
            ts_dir = config['data']['timeseries_dir']
            ts_ext = config['data'].get('timeseries_ext', '.1D')
            if os.path.exists(ts_dir):
                sample_files = [f for f in os.listdir(ts_dir) if f.endswith(ts_ext)][:5]
                logging.info(f"Sample timeseries files: {sample_files}")
                
            raise ValueError("No subjects were loaded successfully. Please check the paths and subject IDs.")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise
    
    logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    logging.info(f"Loaded data for {len(subject_data_dict)} subjects")
    
    # Check for regression target
    regression_target = config['data']['regression_target']
    logging.info(f"Regression target: {regression_target}")
    
    # Log distribution of target variable
    target_values = clinical_data[regression_target]
    logging.info(f"Target distribution: min={target_values.min()}, max={target_values.max()}, mean={target_values.mean():.2f}, std={target_values.std():.2f}")
    
    # Prepare cross-validation
    n_folds = config['training']['n_folds']
    if args.test and n_folds > 3:
        n_folds = 2  # Use fewer folds for testing
        logging.info(f"Test mode: Reducing to {n_folds} folds")
    
    logging.info(f"Preparing {n_folds}-fold cross-validation")
    
    # Create fold indices for stratified k-fold cross-validation
    fold_indices = create_fold_indices(clinical_data, n_folds, config['data'].get('subject_id_col', 'subject'))
    
    # Initialize lists to store results
    fold_results = []
    
    # Train and evaluate model for each fold
    for fold_idx in range(n_folds):
        logging.info(f"===== Starting Fold {fold_idx+1}/{n_folds} =====")
        
        # Create fold directory
        fold_dir = os.path.join(results_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get subject IDs for each set instead of indices
        subject_id_col = config['data'].get('subject_id_col', 'subject')
        all_subjects = clinical_data[subject_id_col].unique()
        
        # Use modulo approach for fixed splits that don't depend on indices
        all_subjects_list = sorted(list(all_subjects))  # Sort for consistent splits
        n_subjects = len(all_subjects_list)
        
        # Calculate number of subjects in each fold
        subjects_per_fold = n_subjects // n_folds
        test_subjects = []
        val_subjects = []
        
        # Deterministic assignment based on fold_idx
        for i, subject in enumerate(all_subjects_list):
            fold_assignment = i % n_folds
            next_fold = (fold_assignment + 1) % n_folds
            
            if fold_assignment == fold_idx:
                test_subjects.append(subject)
            elif next_fold == fold_idx:  # Adjacent fold for validation
                val_subjects.append(subject)
        
        # All other subjects go to training
        train_subjects = [s for s in all_subjects_list if s not in test_subjects and s not in val_subjects]
        
        logging.info(f"Fold {fold_idx}: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects, {len(test_subjects)} test subjects")
        
        # Create custom fold_indices dict with subject lists instead of DataFrame indices
        current_fold_indices = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'test_subjects': test_subjects
        }
        
        # Create data loaders for this fold
        train_loader, val_loader, test_loader = create_data_loaders(
            subject_data_dict, 
            clinical_data,
            config,
            current_fold_indices,
            task_type='regression'
        )
        
        # In test mode, reduce training epochs
        if args.test and config['training']['num_epochs'] > 20:
            config['training']['num_epochs'] = 10
            logging.info(f"Test mode: Using only {config['training']['num_epochs']} epochs")
            
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
    
    # Calculate average results across folds
    avg_test_mse = np.mean([r['test_mse'] for r in fold_results])
    avg_test_mae = np.mean([r['test_mae'] for r in fold_results])
    avg_test_r2 = np.mean([r['test_r2'] for r in fold_results if not np.isnan(r['test_r2'])])
    
    std_test_mse = np.std([r['test_mse'] for r in fold_results])
    std_test_mae = np.std([r['test_mae'] for r in fold_results])
    std_test_r2 = np.std([r['test_r2'] for r in fold_results if not np.isnan(r['test_r2'])])
    
    # Log summary results
    logging.info("===== Cross-Validation Results =====")
    logging.info(f"Average Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
    logging.info(f"Average Test MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
    logging.info(f"Average Test R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")
    
    # Save summary results
    summary_results = {
        'avg_test_mse': float(avg_test_mse),
        'std_test_mse': float(std_test_mse),
        'avg_test_mae': float(avg_test_mae),
        'std_test_mae': float(std_test_mae),
        'avg_test_r2': float(avg_test_r2),
        'std_test_r2': float(std_test_r2),
        'fold_results': fold_results
    }
    
    with open(os.path.join(results_dir, 'summary_results.json'), 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    # Save detailed fold results to CSV
    pd.DataFrame(fold_results).to_csv(os.path.join(results_dir, 'fold_results.csv'), index=False)
    
    logging.info(f"Training completed. Results saved to {results_dir}")

if __name__ == "__main__":
    main() 