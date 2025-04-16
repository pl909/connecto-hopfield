import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # Use TensorDataset for FC features
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
from torch.optim.lr_scheduler import ReduceLROnPlateau # Using ReduceLROnPlateau
from torch.cuda.amp import autocast # Keep for potential future use
from contextlib import nullcontext
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tqdm

# Add src directory to Python path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.utils import load_config, setup_logging, set_seed
    from src.data_loader import load_all_subject_data # Still needed to load windowed data initially
    from src.fc_utils import preprocess_windows_to_fc # New function to compute FC
    from src.fc_models import SimpleMLP # New MLP model
    # Import necessary training functions (adapt if needed)
    from src.training import (
        train_epoch, validate_epoch, EarlyStopping, evaluate_model,
        plot_learning_curves, save_confusion_matrix_plot,
        save_json_metrics, save_training_log
    )
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure you're running from the project root or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)

def get_device(device_name):
    """
    Determines the appropriate device (CPU or CUDA) based on availability.
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

# Note: validate_safely might not be needed for MLP if batch sizes are reasonable.
# We will use validate_epoch directly for simplicity first.

def main(config_path):
    """
    Main function for training an MLP on Functional Connectivity features.
    """
    # Load configuration
    config = load_config(config_path)

    # Create experiment directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = config['experiment_name']
    results_base_dir = config['paths']['results_base_dir']
    experiment_dir = os.path.join(results_base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(experiment_dir, 'training.log')
    setup_logging(log_path)
    logging.info(f"Starting experiment: {experiment_name} (FC-MLP Classification)")
    logging.info(f"Results will be saved to: {experiment_dir}")

    # Save config copy
    config_copy_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f)
    logging.info(f"Configuration saved to {config_copy_path}")

    # Set random seed
    seed = config['training']['seed']
    set_seed(seed)

    # Get device
    device = get_device(config['training']['device'])

    # --- Load Windowed Data and Compute FC Features ---
    phenotypic_file = os.path.join(config['paths']['base_dir'], config['paths']['phenotypic_file'])
    try:
        logging.info("Loading windowed ABIDE data...")
        # 1. Load the original windowed time series data
        X_windows_all, y_windows_all, groups_windows_all, label_encoder = load_all_subject_data(
            config, phenotypic_file, experiment_dir
        )
        num_classes = len(label_encoder.classes_)
        logging.info(f"Loaded {X_windows_all.shape[0]} windows.")

        # 2. Preprocess windows into FC features
        logging.info("Computing Functional Connectivity features...")
        X_fc_all, y_fc_all, groups_fc_all = preprocess_windows_to_fc(
            X_windows_all, y_windows_all, groups_windows_all
        )
        # Free up memory from large windowed array
        del X_windows_all, y_windows_all, groups_windows_all
        
        if X_fc_all.shape[0] == 0:
             logging.error("No valid FC features were generated. Exiting.")
             sys.exit(1)
             
        input_dim = X_fc_all.shape[1] # Dimensionality of the FC feature vector
        logging.info(f"Generated {X_fc_all.shape[0]} FC feature vectors with dimension {input_dim}.")

    except Exception as e:
        logging.error(f"Error during data loading or FC preprocessing: {e}", exc_info=True)
        sys.exit(1)

    # Cross-Validation Setup
    n_folds = config['training']['cv_folds']
    logging.info(f"Starting {n_folds}-fold group cross-validation on FC features (grouped by subject)...")
    group_kfold = GroupKFold(n_splits=n_folds)

    # Storage for metrics
    all_fold_metrics = []
    all_fold_logs = []
    all_fold_cms = []

    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_fc_all, y_fc_all, groups=groups_fc_all)):
        logging.info(f"==== Fold {fold+1}/{n_folds} ====")
        print(f"\n==== Starting fold {fold+1}/{n_folds} ====")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Split FC data
        X_train, y_train = X_fc_all[train_idx], y_fc_all[train_idx]
        X_val, y_val = X_fc_all[val_idx], y_fc_all[val_idx]

        logging.info(f"Train split: {X_train.shape[0]} FC vectors")
        logging.info(f"Validation split: {X_val.shape[0]} FC vectors")
        print(f"Train: {X_train.shape[0]} FC vectors, Validation: {X_val.shape[0]} FC vectors")

        # Create TensorDatasets and DataLoaders for FC data
        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 2),
            pin_memory=config['training'].get('pin_memory', True)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 2),
            pin_memory=config['training'].get('pin_memory', True)
        )

        # Initialize MLP model
        model = SimpleMLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=config['fc_mlp_model']['hidden_layers'],
            dropout_rate=config['fc_mlp_model']['dropout_rate'],
            use_batch_norm=config['fc_mlp_model']['use_batch_norm']
        ).to(device)
        logging.info(f"Initialized SimpleMLP model for fold {fold+1}")

        # Mixed precision setup (usually not needed for MLP, but keep infrastructure)
        use_mixed_precision = config['training'].get('mixed_precision', False)
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None

        # Loss function (can still use class weights if desired)
        class_counts = np.bincount(y_train)
        if len(class_counts) == num_classes: # Ensure counts match classes
            total_samples = len(y_train)
            class_weights = torch.FloatTensor(total_samples / (num_classes * class_counts)).to(device)
            logging.info(f"Using class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            logging.warning(f"Class count mismatch ({len(class_counts)} vs {num_classes}). Using unweighted loss.")
            criterion = nn.CrossEntropyLoss()


        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Scheduler
        scheduler = None
        if config['training'].get('scheduler_type') == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config['training'].get('scheduler_mode', 'min'),
                factor=config['training'].get('scheduler_factor', 0.1),
                patience=config['training'].get('scheduler_patience', 10),
                threshold=config['training'].get('scheduler_threshold', 1e-4),
                min_lr=config['training'].get('scheduler_min_lr', 0),
                verbose=True
            )
            print(f"Using ReduceLROnPlateau scheduler monitoring validation {config['training'].get('scheduler_metric', 'loss')}")
        else:
             print("No scheduler specified or type not recognized.")


        # Early Stopping
        checkpoint_path = os.path.join(fold_dir, 'best_model.pt')
        early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            verbose=True,
            path=checkpoint_path,
            trace_func=logging.info,
            metric=config['training']['early_stopping_metric'] # Use metric from config
        )

        # Training logs
        fold_log = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'grad_norms': [], 'learning_rates': [],
            'batch_losses': [], 'batch_accuracies': [] # Keep batch level if needed
        }

        # Training loop
        for epoch in range(config['training']['num_epochs']):
            print(f"\n==== Epoch {epoch+1}/{config['training']['num_epochs']} ====")

            # Train for one epoch
            # Note: train_epoch might need slight adaptation if it assumes specific model outputs
            # For MLP, it should be fine as it expects logits. Remove scheduler from call.
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                clip_value=config['training'].get('gradient_clip', 1.0),
                scheduler=None # Scheduler stepped per epoch now
            )

            # Validate
            # Use standard validate_epoch as MLP should be less memory intensive
            val_loss, val_accuracy, _, _ = validate_epoch(
                model, val_loader, criterion, device
            )

            # Update log
            fold_log['train_loss'].append(train_metrics['loss'])
            fold_log['train_acc'].append(train_metrics['accuracy'])
            fold_log['val_loss'].append(val_loss)
            fold_log['val_acc'].append(val_accuracy)
            fold_log['grad_norms'].append(np.mean(train_metrics['grad_norms']) if train_metrics['grad_norms'] else 0) # Handle empty list
            fold_log['learning_rates'].extend(train_metrics['learning_rates'])

            # Log progress
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {100 * train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {100 * val_accuracy:.2f}%, "
                f"Grad Norm: {np.mean(train_metrics['grad_norms']) if train_metrics['grad_norms'] else 0:.4f}, " # Handle empty list
                f"LR: {current_lr:.6f}"
            )

            # Step the scheduler based on validation metric
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                metric_to_monitor = config['training'].get('scheduler_metric', 'val_loss')
                if metric_to_monitor == 'val_loss':
                    scheduler.step(val_loss)
                elif metric_to_monitor == 'val_accuracy':
                    scheduler.step(val_accuracy)
                else:
                    logging.warning(f"Scheduler metric '{metric_to_monitor}' not recognized. Stepping with val_loss.")
                    scheduler.step(val_loss) # Default to loss

            # Check early stopping
            early_stopping(val_loss, val_accuracy, model, epoch)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # --- Post-Training ---
        print(f"\n==== Fold {fold+1} training complete, evaluating best model ====")
        log_csv_path = os.path.join(fold_dir, 'epoch_log.csv')
        save_training_log(fold_log, log_csv_path)

        # Load best model
        try:
            model = model.to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logging.info(f"Loaded best model from epoch {early_stopping.best_epoch+1} via {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading best model checkpoint: {e}", exc_info=True)
            logging.info("Using model state from the end of training instead.")

        # Final evaluation
        logging.info("Performing final evaluation on validation set for this fold...")
        final_metrics = evaluate_model(
            model, val_loader, criterion, device, label_encoder.classes_)

        # Save results
        metrics_path = os.path.join(fold_dir, 'final_val_metrics.json')
        save_json_metrics(final_metrics, metrics_path)
        cm_plot_path = os.path.join(fold_dir, 'final_val_confusion_matrix.png')
        save_confusion_matrix_plot(
             final_metrics['confusion_matrix'],
             label_encoder.classes_,
             cm_plot_path
        )

        # Store results for aggregation
        all_fold_metrics.append(final_metrics)
        all_fold_logs.append(fold_log)
        all_fold_cms.append(final_metrics['confusion_matrix'])

        logging.info(f"Completed fold {fold+1}/{n_folds}. Val Acc: {final_metrics['accuracy']:.4f}, Val F1: {final_metrics['f1_score']:.4f}")

    # Aggregate results across folds (reuse existing logic if appropriate)
    # ... (Aggregation logic similar to run_achnn_training.py) ...
    # --- Start Aggregation ---
    logging.info("Aggregating results across all folds...")
    aggregated_dir = os.path.join(experiment_dir, 'aggregated')
    os.makedirs(aggregated_dir, exist_ok=True)

    metric_keys_to_agg = ['accuracy', 'f1_score', 'test_loss']
    agg_metrics = {}
    for metric in metric_keys_to_agg:
        values = [fm[metric] for fm in all_fold_metrics if metric in fm and fm[metric] is not None]
        if values:
             agg_metrics[f'mean_{metric}'] = np.mean(values)
             agg_metrics[f'std_{metric}'] = np.std(values)
        else:
             agg_metrics[f'mean_{metric}'] = np.nan
             agg_metrics[f'std_{metric}'] = np.nan

    agg_metrics_path = os.path.join(aggregated_dir, 'aggregated_metrics.json')
    save_json_metrics(agg_metrics, agg_metrics_path)
    logging.info(f"Saved aggregated metrics to {agg_metrics_path}")

    logging.info("=== Aggregated Performance (Mean ± Std across Folds) ===")
    logging.info(f"  Accuracy: {agg_metrics.get('mean_accuracy', 'N/A'):.4f} ± {agg_metrics.get('std_accuracy', 'N/A'):.4f}")
    logging.info(f"  F1 Score: {agg_metrics.get('mean_f1_score', 'N/A'):.4f} ± {agg_metrics.get('std_f1_score', 'N/A'):.4f}")
    logging.info(f"  Test Loss: {agg_metrics.get('mean_test_loss', 'N/A'):.4f} ± {agg_metrics.get('std_test_loss', 'N/A'):.4f}")

    # Plot average learning curves (needs adaptation if plot_learning_curves expects dict)
    try:
        curves_path = os.path.join(aggregated_dir, 'aggregated_learning_curves.png')
        # This function might need adjustment if it expects specific dict keys not present in fold_log now
        plot_learning_curves(
            [log['train_loss'] for log in all_fold_logs],
            [log['val_loss'] for log in all_fold_logs],
            [log['train_acc'] for log in all_fold_logs],
            [log['val_acc'] for log in all_fold_logs],
            [log['grad_norms'] for log in all_fold_logs],
            [log['learning_rates'] for log in all_fold_logs], # Pass list of lists
            curves_path
        )
    except Exception as e:
        logging.error(f"Could not plot aggregated learning curves: {e}", exc_info=True)

    # Aggregate confusion matrix
    if all_fold_cms:
        agg_cm = np.sum(all_fold_cms, axis=0)
        cm_plot_path = os.path.join(aggregated_dir, 'aggregated_confusion_matrix.png')
        save_confusion_matrix_plot(
            agg_cm,
            label_encoder.classes_,
            cm_plot_path,
            title='Aggregated Confusion Matrix (Summed over Folds)'
        )
    # --- End Aggregation ---

    logging.info(f"Experiment '{experiment_name}' completed successfully.")
    logging.info(f"Aggregated results saved to {aggregated_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FC-MLP model for ABIDE ASD vs TDC classification")
    parser.add_argument("config", help="Path to configuration YAML file (e.g., configs/fc_mlp_config.yaml)")
    args = parser.parse_args()
    main(args.config)
