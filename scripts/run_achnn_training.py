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

# Add src directory to Python path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.utils import load_config, setup_logging, set_seed
    from src.data_loader import load_all_subject_data, FMRIWindowDataset
    from src.models import ACHNN
    from src.training import train_epoch, evaluate, EarlyStopping, save_training_log, save_json_metrics
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure you're running from the project root or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)

def plot_learning_curves(fold_logs, output_path):
    """
    Plots average training and validation loss/accuracy across folds.
    
    Args:
        fold_logs: List of dictionaries containing training/validation metrics per fold
        output_path: Where to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Extract metrics
    train_losses = [log['train_loss'] for log in fold_logs]
    val_losses = [log['val_loss'] for log in fold_logs]
    train_accs = [log['train_acc'] for log in fold_logs]
    val_accs = [log['val_acc'] for log in fold_logs]
    
    # Find maximum number of epochs across all folds
    max_epochs = max(len(losses) for losses in train_losses)
    
    # Pad shorter sequences with NaN for consistent averaging
    train_losses_padded = [losses + [np.nan] * (max_epochs - len(losses)) for losses in train_losses]
    val_losses_padded = [losses + [np.nan] * (max_epochs - len(losses)) for losses in val_losses]
    train_accs_padded = [accs + [np.nan] * (max_epochs - len(accs)) for accs in train_accs]
    val_accs_padded = [accs + [np.nan] * (max_epochs - len(accs)) for accs in val_accs]
    
    # Calculate mean metrics, ignoring NaN values
    mean_train_loss = np.nanmean(train_losses_padded, axis=0)
    mean_val_loss = np.nanmean(val_losses_padded, axis=0)
    mean_train_acc = np.nanmean(train_accs_padded, axis=0)
    mean_val_acc = np.nanmean(val_accs_padded, axis=0)
    
    epochs = range(1, max_epochs + 1)
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mean_train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, mean_val_loss, 'r-', label='Validation Loss')
    plt.title('Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mean_train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, mean_val_acc, 'r-', label='Validation Accuracy')
    plt.title('Average Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Learning curves saved to {output_path}")

def save_confusion_matrix_plot(cm, class_names, output_path, title='Confusion Matrix'):
    """
    Creates and saves a confusion matrix plot.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: Names of the classes
        output_path: Where to save the plot
        title: Title of the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")

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
    Main function that runs the full training and cross-validation process.
    
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
    logging.info(f"Starting experiment: {experiment_name}")
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
    
    # Load subject IDs from QC filter output
    included_subjects_file = os.path.join(results_base_dir, 'included_subjects.txt')
    try:
        with open(included_subjects_file, 'r') as f:
            included_subjects = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(included_subjects)} subjects from {included_subjects_file}")
    except FileNotFoundError:
        logging.error(f"Included subjects file not found: {included_subjects_file}")
        logging.error("Please run the QC filtering script first.")
        sys.exit(1)
    
    # Load and prepare data
    try:
        logging.info("Loading and preparing data...")
        X_all, y_all, groups_all, label_encoder = load_all_subject_data(
            config, included_subjects_file, experiment_dir)
        
        num_classes = len(label_encoder.classes_)
        if 'num_classes' in config['achnn_model'] and num_classes != config['achnn_model']['num_classes']:
            logging.warning(f"Number of classes in data ({num_classes}) differs from config"
                          f" ({config['achnn_model']['num_classes']}). Updating config.")
            config['achnn_model']['num_classes'] = num_classes
        elif 'num_classes' not in config['achnn_model']:
            logging.info(f"Setting num_classes in config to {num_classes}")
            config['achnn_model']['num_classes'] = num_classes
        
        logging.info(f"Data loaded: {X_all.shape[0]} samples, {num_classes} classes")
        logging.info(f"Class labels: {label_encoder.classes_.tolist()}")
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(1)
    
    # Cross-Validation Setup
    n_folds = config['training']['cv_folds']
    logging.info(f"Starting {n_folds}-fold cross-validation...")
    
    # Initialize GroupKFold
    group_kfold = GroupKFold(n_splits=n_folds)
    
    # Storage for metrics and logs
    all_fold_metrics = []
    all_fold_logs = []
    all_fold_cms = []
    
    # Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_all, y_all, groups=groups_all)):
        logging.info(f"==== Fold {fold+1}/{n_folds} ====")
        
        # Create fold directory
        fold_dir = os.path.join(experiment_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Split data
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        
        logging.info(f"Train split: {X_train.shape[0]} samples")
        logging.info(f"Validation split: {X_val.shape[0]} samples")
        
        # Create datasets and dataloaders
        train_dataset = FMRIWindowDataset(X_train, y_train)
        val_dataset = FMRIWindowDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == 'cuda')
        )
        
        # Initialize model
        model = ACHNN(config, num_classes=num_classes).to(device)
        logging.info(f"Initialized ACHNN model with {num_classes} output classes")
        
        # Loss function, optimizer, and early stopping
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        checkpoint_path = os.path.join(fold_dir, 'best_model.pt')
        early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            verbose=True,
            path=checkpoint_path,
            trace_func=logging.info
        )
        
        # Training logs for this fold
        fold_log = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(config['training']['epochs']):
            # Train for one epoch
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate on validation set
            val_metrics, val_cm, val_cm_plot = evaluate(
                model, val_loader, criterion, device, label_encoder)
            
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # Update log
            fold_log['train_loss'].append(train_loss)
            fold_log['train_acc'].append(train_acc)
            fold_log['val_loss'].append(val_loss)
            fold_log['val_acc'].append(val_acc)
            
            # Log progress
            logging.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save training log for this fold
        log_path = os.path.join(fold_dir, 'train_log.csv')
        save_training_log(fold_log, log_path)
        
        # Load best model
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            logging.info(f"Loaded best model from {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error loading best model: {e}")
            logging.info("Using current model instead")
        
        # Final evaluation
        final_metrics, final_cm, final_cm_plot = evaluate(
            model, val_loader, criterion, device, label_encoder)
        
        # Save metrics and confusion matrix
        metrics_path = os.path.join(fold_dir, 'val_metrics.json')
        save_json_metrics(final_metrics, metrics_path)
        
        if final_cm_plot is not None:
            cm_plot_path = os.path.join(fold_dir, 'val_confusion_matrix.png')
            final_cm_plot.savefig(cm_plot_path, dpi=150)
            plt.close()
            logging.info(f"Saved confusion matrix plot to {cm_plot_path}")
        
        # Store metrics for aggregation
        all_fold_metrics.append(final_metrics)
        all_fold_logs.append(fold_log)
        all_fold_cms.append(final_cm)
        
        logging.info(f"Completed fold {fold+1}/{n_folds}")
    
    # Aggregate results across folds
    logging.info("Aggregating results across all folds...")
    
    # Create aggregated directory
    aggregated_dir = os.path.join(experiment_dir, 'aggregated')
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Calculate mean/std of metrics across folds
    agg_metrics = {}
    for metric in all_fold_metrics[0].keys():
        if metric in ['loss', 'accuracy', 'f1_weighted']:
            values = [fold_metric[metric] for fold_metric in all_fold_metrics]
            agg_metrics[f'mean_{metric}'] = np.mean(values)
            agg_metrics[f'std_{metric}'] = np.std(values)
    
    # Save aggregated metrics
    agg_metrics_path = os.path.join(aggregated_dir, 'mean_metrics.json')
    save_json_metrics(agg_metrics, agg_metrics_path)
    logging.info(f"Saved aggregated metrics to {agg_metrics_path}")
    
    # Log aggregated performance
    logging.info("=== Aggregated Performance ===")
    for metric, value in agg_metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Plot and save learning curves
    curves_path = os.path.join(aggregated_dir, 'learning_curves.png')
    plot_learning_curves(all_fold_logs, curves_path)
    
    # Aggregate and save confusion matrix
    agg_cm = np.sum(all_fold_cms, axis=0)
    cm_plot_path = os.path.join(aggregated_dir, 'confusion_matrix.png')
    save_confusion_matrix_plot(
        agg_cm, 
        label_encoder.classes_, 
        cm_plot_path, 
        title='Aggregated Confusion Matrix'
    )
    
    logging.info(f"Experiment '{experiment_name}' completed successfully.")
    logging.info(f"Results saved to {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACHNN model with cross-validation")
    parser.add_argument("config", help="Path to configuration YAML file")
    args = parser.parse_args()
    
    main(args.config) 