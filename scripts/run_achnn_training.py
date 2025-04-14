import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd
import os
import sys
import time
import json
import argparse
import logging
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Add src directory to Python path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.utils import load_config, setup_logging, set_seed
    from src.data_loader import load_all_subject_data, FMRIWindowDataset 
    from src.models import ACHNN
    # Corrected import list for src/training.py
    from src.training import ( 
        train_epoch, validate_epoch, EarlyStopping, evaluate_model, 
        plot_learning_curves, save_confusion_matrix_plot, 
        save_json_metrics, save_training_log, # Added missing functions here
        analyze_hopfield_attention, visualize_latent_space # Keep these if needed elsewhere, though not used directly in run_training.py
    )
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure you're running from the project root or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)

# Removed plot_learning_curves and save_confusion_matrix_plot as they are now in src/training.py

def get_device(device_name):
    """
    Determines the appropriate device (CPU or CUDA) based on availability.
    
    Args:
        device_name: Requested device name ('cuda' or 'cpu')
        
    Returns:
        torch.device object
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if device_name == 'cuda':
            logging.warning("CUDA requested but not available. Using CPU instead.")
        device = torch.device('cpu')
        logging.info("Using CPU.")
    return device

def main(config_path):
    """
    Main function that runs the full training and cross-validation process for ABIDE ASD vs TDC.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create experiment directory with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = config['experiment_name']
    results_base_dir = config['paths']['results_base_dir']
    experiment_dir = os.path.join(results_base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(experiment_dir, 'training.log')
    setup_logging(log_path)
    logging.info(f"Starting experiment: {experiment_name} (ABIDE ASD vs TDC Classification)")
    logging.info(f"Results will be saved to: {experiment_dir}")
    
    # Save config copy
    config_copy_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f)
    logging.info(f"Configuration saved to {config_copy_path}")
    
    # Set random seed for reproducibility
    seed = config['training']['seed']
    set_seed(seed)
    
    # Get device (CPU or CUDA)
    device = get_device(config['training']['device'])
    
    # --- Load and prepare ABIDE data ---
    phenotypic_file = os.path.join(config['paths']['base_dir'], config['paths']['phenotypic_file'])
    try:
        logging.info("Loading and preparing ABIDE data...")
        # The load_all_subject_data function now handles reading the CSV and filtering internally
        X_all, y_all, groups_all, label_encoder = load_all_subject_data(
            config, phenotypic_file, experiment_dir # Pass pheno file path directly
        )
        
        # Ensure num_classes is correctly set for binary classification
        num_classes = len(label_encoder.classes_)
        if num_classes != 2:
             logging.warning(f"Expected 2 classes (ASD/TDC) but found {num_classes}. Check DX_GROUP filtering.")
             # Update config just in case, though it should be 2
             config['achnn_model']['num_classes'] = num_classes
        else:
             config['achnn_model']['num_classes'] = 2 # Explicitly set for clarity
             logging.info(f"Confirmed binary classification task with {num_classes} classes.")

        # Ensure num_regions matches the atlas used (e.g., CC200)
        expected_regions = config['data']['num_regions']
        actual_regions = X_all.shape[2]
        if actual_regions != expected_regions:
            logging.error(f"Mismatch between config num_regions ({expected_regions}) and data features ({actual_regions}).")
            sys.exit(1)

        logging.info(f"Data loaded: {X_all.shape[0]} total windows across {len(np.unique(groups_all))} subjects.")
        logging.info(f"Class labels (encoded): {np.unique(y_all)}")
        logging.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    except Exception as e:
        logging.error(f"Error loading or preparing ABIDE data: {e}", exc_info=True)
        sys.exit(1)
    
    # Cross-Validation Setup
    n_folds = config['training']['cv_folds']
    logging.info(f"Starting {n_folds}-fold group cross-validation (grouped by subject)...")
    
    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_folds)
    
    # Storage for metrics and logs
    all_fold_metrics = []
    all_fold_logs = [] # Store epoch logs for each fold
    all_fold_cms = []
    
    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_all, y_all, groups=groups_all)):
        logging.info(f"==== Fold {fold+1}/{n_folds} ====")
        
        # Create fold directory
        fold_dir = os.path.join(experiment_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data for this fold
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        logging.info(f"Train split: {X_train.shape[0]} windows from {len(np.unique(groups_all[train_idx]))} subjects")
        logging.info(f"Validation split: {X_val.shape[0]} windows from {len(np.unique(groups_all[val_idx]))} subjects")
        
        # Create datasets and dataloaders
        train_dataset = FMRIWindowDataset(X_train, y_train)
        val_dataset = FMRIWindowDataset(X_val, y_val)
        
        # Calculate class weights to handle imbalance
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        logging.info(f"Using class weights: {class_weights}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=16,  # Use all 16 CPUs for data loading
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True,  # Keep workers alive between batches
            prefetch_factor=2  # Prefetch batches for better throughput
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=8,  # Use fewer workers for validation
            pin_memory=True,  # Pin memory for faster GPU transfer
            persistent_workers=True,  # Keep workers alive between batches
            prefetch_factor=2  # Prefetch batches for better throughput
        )
        
        # Initialize model (make sure num_classes=2)
        model = ACHNN(config, num_classes=2)
        # Use DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        model = model.to(device)
        logging.info(f"Initialized ACHNN model for fold {fold+1}")
        
        # Enable CUDA optimizations if available
        if device.type == 'cuda':
            # Set autocast for mixed precision training (faster and uses less memory)
            use_mixed_precision = config['training'].get('mixed_precision', False)  # Default to False for stability
            scaler = torch.amp.GradScaler() if use_mixed_precision else None
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            logging.info(f"CUDA optimizations enabled: mixed precision={use_mixed_precision} and cuDNN benchmark=True")
        else:
            use_mixed_precision = False
            scaler = None
        
        # Loss function with class weights
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Set up optimizer with layer-specific parameters
        # Apply higher learning rate to classifier and lower rate to the rest
        base_lr = config['training']['learning_rate']
        params = [
            {'params': model.module.hopfield.parameters() if hasattr(model, 'module') else model.hopfield.parameters(), 
             'lr': base_lr * 1.5},  # Higher LR for Hopfield
            {'params': model.module.classifier.parameters() if hasattr(model, 'module') else model.classifier.parameters(), 
             'lr': base_lr * 2.0},  # Higher LR for classifier
            {'params': model.module.embed.parameters() if hasattr(model, 'module') else model.embed.parameters(), 
             'lr': base_lr},  # Base LR for embedding
            {'params': model.module.transformer_encoder.parameters() if hasattr(model, 'module') else model.transformer_encoder.parameters(), 
             'lr': base_lr * 0.8},  # Slightly lower LR for transformer
        ]
        
        # AdamW optimizer with parameter-specific learning rates
        optimizer = optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use OneCycleLR scheduler with warmup
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config['training']['num_epochs']
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[base_lr * 1.5, base_lr * 2.0, base_lr, base_lr * 0.8],
            total_steps=total_steps,
            pct_start=0.1,  # Use first 10% of training for warmup
            div_factor=25,  # Initial lr is max_lr/25
            final_div_factor=10000,  # Final lr is max_lr/10000
            anneal_strategy='cos'  # Use cosine annealing
        )
        
        checkpoint_path = os.path.join(fold_dir, 'best_model.pt')
        early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            verbose=True,
            path=checkpoint_path,
            trace_func=logging.info # Log early stopping messages
        )
        
        # Training logs for this fold
        fold_log = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [] # Store val accuracy per epoch
        }
        
        # Training loop for the fold
        for epoch in range(config['training']['num_epochs']):
            # Train for one epoch
            if device.type == 'cuda' and use_mixed_precision:
                # Use mixed precision training
                with torch.amp.autocast('cuda'):
                    train_loss, train_acc = train_epoch(
                        model, 
                        train_loader, 
                        criterion, 
                        optimizer, 
                        device, 
                        scaler=scaler,
                        clip_value=config['training'].get('gradient_clip', 1.0),
                        scheduler=scheduler
                    ) 
                # Evaluate on validation set with mixed precision
                with torch.amp.autocast('cuda'):
                    val_loss, val_accuracy, _, _ = validate_epoch(model, val_loader, criterion, device)
            else:
                # Regular training without mixed precision
                train_loss, train_acc = train_epoch(
                    model, 
                    train_loader, 
                    criterion, 
                    optimizer, 
                    device,
                    scaler=None,  # Explicitly set to None when not using mixed precision
                    clip_value=config['training'].get('gradient_clip', 1.0),
                    scheduler=scheduler
                ) 
                val_loss, val_accuracy, _, _ = validate_epoch(model, val_loader, criterion, device)
            
            # Update log
            fold_log['train_loss'].append(train_loss)
            fold_log['val_loss'].append(val_loss)
            fold_log['val_acc'].append(val_accuracy) # Log val accuracy
            
            # Don't update scheduler here - it's now updated in the train_epoch function
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log progress
            logging.info(
                f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " # Log train accuracy
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Check early stopping based on validation loss
            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save training log (epoch-wise metrics) for this fold
        log_csv_path = os.path.join(fold_dir, 'epoch_log.csv')
        save_training_log(fold_log, log_csv_path) # Use updated function name
        
        # Load best model from checkpoint
        try:
            # Ensure model is on the correct device before loading state dict
            model = model.to(device) 
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"Loaded best model from epoch {early_stopping.best_epoch+1} via {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading best model checkpoint: {e}")
            logging.info("Using model state from the end of training instead.")
        
        # Final evaluation on the validation set for this fold
        logging.info("Performing final evaluation on validation set for this fold...")
        final_metrics = evaluate_model( # Use updated function name
            model, val_loader, criterion, device, label_encoder.classes_)
        
        # Save final metrics and confusion matrix for the fold
        metrics_path = os.path.join(fold_dir, 'final_val_metrics.json')
        save_json_metrics(final_metrics, metrics_path) # Use updated function name
        
        cm_plot_path = os.path.join(fold_dir, 'final_val_confusion_matrix.png')
        save_confusion_matrix_plot( # Use updated function name
             final_metrics['confusion_matrix'], 
             label_encoder.classes_, 
             cm_plot_path
        )
        
        # Store results for aggregation
        all_fold_metrics.append(final_metrics)
        all_fold_logs.append(fold_log) # Store epoch logs
        all_fold_cms.append(final_metrics['confusion_matrix'])
        
        logging.info(f"Completed fold {fold+1}/{n_folds}. Val Acc: {final_metrics['accuracy']:.4f}, Val F1: {final_metrics['f1_score']:.4f}")
    
    # Aggregate results across folds
    logging.info("Aggregating results across all folds...")
    
    # Create aggregated directory
    aggregated_dir = os.path.join(experiment_dir, 'aggregated')
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Calculate mean/std of final validation metrics across folds
    agg_metrics = {}
    # Use keys from the last fold's metrics dict as reference
    metric_keys_to_agg = ['accuracy', 'f1_score', 'test_loss'] # Use 'f1_score' if weighted is default
    
    for metric in metric_keys_to_agg:
        values = [fold_metric[metric] for fold_metric in all_fold_metrics if metric in fold_metric]
        if values: # Check if list is not empty
             agg_metrics[f'mean_{metric}'] = np.mean(values)
             agg_metrics[f'std_{metric}'] = np.std(values)
        else:
             agg_metrics[f'mean_{metric}'] = np.nan
             agg_metrics[f'std_{metric}'] = np.nan

    # Save aggregated metrics
    agg_metrics_path = os.path.join(aggregated_dir, 'aggregated_metrics.json')
    save_json_metrics(agg_metrics, agg_metrics_path)
    logging.info(f"Saved aggregated metrics to {agg_metrics_path}")
    
    # Log aggregated performance
    logging.info("=== Aggregated Performance (Mean ± Std across Folds) ===")
    logging.info(f"  Accuracy: {agg_metrics.get('mean_accuracy', 'N/A'):.4f} ± {agg_metrics.get('std_accuracy', 'N/A'):.4f}")
    logging.info(f"  F1 Score: {agg_metrics.get('mean_f1_score', 'N/A'):.4f} ± {agg_metrics.get('std_f1_score', 'N/A'):.4f}")
    logging.info(f"  Val Loss: {agg_metrics.get('mean_test_loss', 'N/A'):.4f} ± {agg_metrics.get('std_test_loss', 'N/A'):.4f}")

    # Plot and save average learning curves
    curves_path = os.path.join(aggregated_dir, 'aggregated_learning_curves.png')
    plot_learning_curves(all_fold_logs, curves_path) # Use updated function name
    
    # Aggregate and save confusion matrix
    if all_fold_cms:
        agg_cm = np.sum(all_fold_cms, axis=0)
        cm_plot_path = os.path.join(aggregated_dir, 'aggregated_confusion_matrix.png')
        save_confusion_matrix_plot( # Use updated function name
            agg_cm, 
            label_encoder.classes_, 
            cm_plot_path, 
            title='Aggregated Confusion Matrix (Summed over Folds)'
        )
    
    logging.info(f"Experiment '{experiment_name}' completed successfully.")
    logging.info(f"Aggregated results saved to {aggregated_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACHNN model for ABIDE ASD vs TDC classification with cross-validation")
    parser.add_argument("config", help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)