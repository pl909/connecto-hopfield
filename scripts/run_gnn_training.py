#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

# Add the src directory to the path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from training import train_epoch, validate_epoch, EarlyStopping
from utils import set_seed
from gnn_models import create_gnn_model
from gnn_utils import GraphDataLoader
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device):
    """Evaluate model performance on the test dataset.
    
    Args:
        model (nn.Module): The trained model
        dataloader (DataLoader): Test dataloader
        device (torch.device): Device to use for evaluation
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    # Print start of evaluation
    print(f"Starting model evaluation on {len(dataloader.dataset)} samples...")
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Get total number of batches for progress reporting
    total_batches = len(dataloader)
    
    # Set up for mixed precision if using CUDA
    if device.type == 'cuda':
        from torch.cuda.amp import autocast
        context_manager = autocast()
    else:
        import contextlib
        context_manager = contextlib.nullcontext()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Evaluating batch {batch_idx}/{total_batches}...")
            
            # Move data to device
            data = data.to(device)
            targets = data.y
            
            # Forward pass with mixed precision
            with context_manager:
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            # Accumulate test loss
            test_loss += loss.item()
            
            # Get predictions
            _, predictions = torch.max(outputs, 1)
            
            # Update counts
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            # Save predictions and targets for metrics
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Free memory
            del outputs, predictions, targets, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Calculate metrics only if we collected predictions
    if len(all_preds) > 0 and len(all_targets) > 0:
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        avg_test_loss = test_loss / len(dataloader)
        
        print(f"Evaluation complete:")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'test_loss': avg_test_loss,
            'confusion_matrix': conf_matrix
        }
    else:
        logger.warning("No predictions or targets collected during evaluation")
        return {
            'accuracy': 0,
            'f1_score': 0,
            'test_loss': float('inf'),
            'confusion_matrix': np.zeros((1, 1))
        }


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_confusion_matrix(confusion_mat, class_names, save_path):
    """Save confusion matrix as an image.
    
    Args:
        confusion_mat (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save the image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_training_history(history, save_path):
    """Save training history as an image.
    
    Args:
        history (dict): Dictionary with training metrics
        save_path (str): Path to save the image
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(config_path):
    """Main function for training a GNN model.
    
    Args:
        config_path (str): Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config['experiment']['name']}_{timestamp}"
    output_dir = os.path.join(config['experiment']['results_path'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging to file
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set seed for reproducibility
    seed = config['training']['seed']
    set_seed(seed)
    
    # Set device
    device_name = config['training']['device']
    device = torch.device(device_name)
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    data_loader = GraphDataLoader(
        data_path=config['experiment']['data_path'],
        batch_size=config['training']['batch_size'],
        threshold=config['qc']['threshold'],
        node_feature_type=config['model'].get('node_features', 'degree'),
        normalize_method=config['model'].get('edge_normalization', 'none'),
        test_size=config['data']['test_size'],
        val_size=config['data']['validation_size'],
        random_state=seed
    )
    
    # Load and split the data
    # We need to implement data loading logic since it's not in GraphDataLoader
    from data_loader import load_abide_timeseries
    
    # Load the timeseries data and prepare matrices
    timeseries_data, labels = load_abide_timeseries(
        base_dir=config['paths']['base_dir'],
        phenotypic_file=config['paths']['phenotypic_file'],
        timeseries_dir=config['paths']['regional_timeseries_dir'],
        included_sites=config['qc']['included_sites'],
        num_regions=config['data']['num_regions'],
        tr=config['data']['tr'],
        window_length=config['data']['seq_len'],
        window_step=config['data']['window_step']
    )
    
    # Convert timeseries to connectivity matrices
    from fc_utils import compute_fc_matrices
    all_fc_matrices, bad_subject_indices = compute_fc_matrices(timeseries_data)

    # Filter out subjects with zero variance
    if bad_subject_indices:
        logger.warning(f"Found {len(bad_subject_indices)} subjects with zero variance regions. Excluding them from the dataset.")
        logger.info(f"Indices of excluded subjects: {bad_subject_indices}")
        
        # Create a mask for valid subjects
        valid_subject_mask = np.ones(len(all_fc_matrices), dtype=bool)
        valid_subject_mask[bad_subject_indices] = False
        
        # Filter matrices and labels
        fc_matrices = all_fc_matrices[valid_subject_mask]
        labels = labels[valid_subject_mask]
        
        logger.info(f"Dataset size reduced from {len(all_fc_matrices)} to {len(fc_matrices)} subjects after filtering.")
    else:
        logger.info("No subjects with zero variance regions found. Using the full dataset.")
        fc_matrices = all_fc_matrices # Use all matrices if none were bad
    
    # Ensure we still have data left after filtering
    if len(fc_matrices) == 0:
        logger.error("No subjects remaining after filtering for zero variance. Cannot proceed.")
        return # Or raise an error

    # Load filtered data into our data loader
    data_loader.load_data(fc_matrices, labels)
    
    # Get the data loaders
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    
    # Set number of classes based on unique labels
    num_classes = len(np.unique(labels))
    
    # Determine input dimension based on the first graph in the train loader
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.size(1)
    
    # Create model
    logger.info("Creating GNN model...")
    hidden_dim = config['model']['hidden_dim']
    
    # Get model-specific parameters
    model_params = {
        'num_layers': config['model'].get('num_layers', 2),
        'dropout': config['model'].get('dropout', 0.5),
        'pool_method': config['model'].get('pool_method', 'mean'),
        'use_edge_attr': config['model'].get('use_edge_attr', True),
        'batch_norm': config['model'].get('batch_norm', True)
    }
    
    # Add model-specific parameters
    if config['model']['type'] == 'gat':
        model_params.update({
            'heads': config['model'].get('heads', 4),
            'concat': config['model'].get('concat', True)
        })
    elif config['model']['type'] == 'dynamic_edge':
        model_params.update({
            'edge_dim': config['model'].get('edge_dim', 1),
            'conv_type': config['model'].get('conv_type', 'gcn'),
            'update_edges': config['model'].get('update_edges', True)
        })
    
    # Create model
    model = create_gnn_model(
        model_type=config['model']['type'],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        model_params=model_params
    )
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params} parameters ({trainable_params} trainable)")
    
    # Set up optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Set up learning rate scheduler
    scheduler = None
    if config['training'].get('use_lr_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('lr_factor', 0.1),
            patience=config['training'].get('lr_patience', 10),
            verbose=True
        )
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 20),
        verbose=True,
        delta=config['training'].get('early_stopping_delta', 0.0),
        path=os.path.join(output_dir, 'best_model.pt')
    )
    
    # Set up TensorBoard writer
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize training variables
    num_epochs = config['training']['num_epochs']
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Check if FP16 precision should be used
    use_amp = config['training'].get('use_mixed_precision', False) and device.type == 'cuda'
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        logger.info("Using mixed precision training")
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        if use_amp:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler=scaler
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
        
        # Validation phase
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Write to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save training history plot
    save_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Load the best model
    logger.info("Loading best model for evaluation")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Log test metrics
    logger.info(f"Test Loss: {test_metrics['test_loss']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    # Write test metrics to file
    with open(os.path.join(output_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump({
            'test_loss': float(test_metrics['test_loss']),
            'test_accuracy': float(test_metrics['accuracy']),
            'test_f1_score': float(test_metrics['f1_score'])
        }, f)
    
    # Save confusion matrix
    class_names = [str(i) for i in range(num_classes)]
    save_confusion_matrix(
        test_metrics['confusion_matrix'],
        class_names,
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a GNN model on brain data')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    main(args.config) 