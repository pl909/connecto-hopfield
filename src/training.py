import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
import os
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                                   Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        try:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
        except Exception as e:
            self.trace_func(f"Error saving model checkpoint to {self.path}: {e}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
    
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
    
    epoch_loss = epoch_loss / len(dataloader.dataset)
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device):
    """
    Validate the model on the validation set.
    
    Args:
        model: The model to validate
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
    
    Returns:
        tuple: (val_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            val_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    val_loss = val_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    
    return val_loss, accuracy, np.array(all_preds), np.array(all_targets)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, checkpoint_dir, early_stopping_patience=10):
    """
    Train the model with early stopping.
    
    Args:
        model: The model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs to train for
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save model checkpoints
        early_stopping_patience: Number of epochs to wait for improvement before stopping
    
    Returns:
        dict: Training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        verbose=True, 
        path=checkpoint_path,
        trace_func=logging.info
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_predictions': [],
        'val_targets': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_predictions'].append(val_preds)
        history['val_targets'].append(val_targets)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}: '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Load best model
    try:
        best_model_state = torch.load(checkpoint_path)
        model.load_state_dict(best_model_state)
        logging.info(f'Loaded best model from {checkpoint_path}')
    except Exception as e:
        logging.error(f"Error loading best model: {e}")
    
    return history


def evaluate_model(model, test_loader, criterion, device, class_names=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        class_names: List of class names for confusion matrix labels
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_hopfield_attention = []
    
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, hopfield_attn, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            test_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Store Hopfield attention
            if hopfield_attn is not None:
                all_hopfield_attention.append(hopfield_attn.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    
    # Compile Hopfield attention
    if all_hopfield_attention:
        all_hopfield_attention = np.concatenate(all_hopfield_attention, axis=0)
    
    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'hopfield_attention': all_hopfield_attention if all_hopfield_attention else None
    }
    
    logging.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    
    return metrics


def save_confusion_matrix_plot(confusion_matrix, class_names, output_path):
    """
    Create and save a confusion matrix plot.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize by row (true labels)
    cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Confusion matrix saved to {output_path}')


def plot_learning_curves(train_loss, val_loss, val_accuracy, output_path):
    """
    Plot and save learning curves.
    
    Args:
        train_loss: List of training losses per epoch
        val_loss: List of validation losses per epoch
        val_accuracy: List of validation accuracies per epoch
        output_path: Path to save the plot
    """
    epochs = range(1, len(train_loss) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy curve
    ax2.plot(epochs, val_accuracy, 'g-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Learning curves saved to {output_path}')


def analyze_hopfield_attention(model, loader, class_names, output_dir, device):
    """
    Analyze the Hopfield attention patterns for each class.
    
    Args:
        model: Trained model
        loader: DataLoader
        class_names: List of class names
        output_dir: Directory to save the analysis
        device: Device to run the model on
        
    Returns:
        dict: Attention analysis results
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect attention patterns by class
    attention_by_class = {class_name: [] for class_name in class_names}
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            _, hopfield_attn, _ = model(inputs)
            
            if hopfield_attn is not None:
                # For each sample in the batch
                for i, target in enumerate(targets):
                    class_name = class_names[target.item()]
                    # Get attention pattern for this sample
                    # If model has multiple Hopfield heads, average over heads
                    if hopfield_attn.dim() == 3:  # [batch, num_heads, num_patterns]
                        sample_attn = hopfield_attn[i].mean(dim=0).cpu().numpy()
                    else:  # [batch, num_patterns]
                        sample_attn = hopfield_attn[i].cpu().numpy()
                    attention_by_class[class_name].append(sample_attn)
    
    # Average attention patterns for each class
    avg_attention_by_class = {}
    for class_name, attention_list in attention_by_class.items():
        if attention_list:
            avg_attention = np.mean(np.stack(attention_list), axis=0)
            avg_attention_by_class[class_name] = avg_attention
            
            # Log the average attention values for this class
            logging.info(f"Average attention for class '{class_name}':")
            for i, val in enumerate(avg_attention):
                logging.info(f"  Pattern {i+1}: {val:.4f}")
    
    # Create a heatmap of class-pattern attention
    num_patterns = model.hopfield_num_stored_patterns
    attention_matrix = np.zeros((len(class_names), num_patterns))
    
    for i, class_name in enumerate(class_names):
        if class_name in avg_attention_by_class:
            attention_matrix[i] = avg_attention_by_class[class_name]
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(attention_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[f'Pattern {i+1}' for i in range(num_patterns)],
                yticklabels=class_names)
    plt.xlabel('Hopfield Stored Patterns')
    plt.ylabel('Class')
    plt.title('Average Hopfield Attention by Class')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'hopfield_attention_by_class.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Hopfield attention analysis saved to {output_path}')
    
    results = {
        'attention_by_class': avg_attention_by_class,
        'attention_matrix': attention_matrix
    }
    
    # Save the numeric data for further analysis
    np.save(os.path.join(output_dir, 'hopfield_attention_matrix.npy'), attention_matrix)
    
    return results


def visualize_latent_space(model, loader, class_names, output_dir, device, 
                           method='tsne', perplexity=30, n_components=2):
    """
    Visualize the latent space of the model using dimensionality reduction.
    
    Args:
        model: Trained model
        loader: DataLoader
        class_names: List of class names
        output_dir: Directory to save the visualizations
        device: Device to run the model on
        method: Dimensionality reduction method ('pca' or 'tsne')
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components for dimensionality reduction
        
    Returns:
        dict: Visualization results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract latent representations
    latent_vectors, labels = model.extract_latent_representations(loader, device)
    
    # Apply dimensionality reduction
    logging.info(f"Applying {method.upper()} dimensionality reduction...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_vectors = reducer.fit_transform(latent_vectors)
        explained_var = reducer.explained_variance_ratio_
        logging.info(f"PCA explained variance: {explained_var.sum():.4f}")
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
        reduced_vectors = reducer.fit_transform(latent_vectors)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Create a colormap with one color per class
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    
    # Plot each class separately to create a legend
    for i, class_name in enumerate(class_names):
        idx = labels == i
        if np.any(idx):
            plt.scatter(reduced_vectors[idx, 0], reduced_vectors[idx, 1], 
                      c=[cmap(i)], label=class_name, alpha=0.7, edgecolors='none')
    
    plt.legend(loc='best')
    plt.title(f'Latent Space Visualization ({method.upper()})')
    
    output_path = os.path.join(output_dir, f'latent_space_{method.lower()}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Latent space visualization saved to {output_path}')
    
    # Save the reduced vectors and labels for further analysis
    np.save(os.path.join(output_dir, 'latent_vectors_reduced.npy'), reduced_vectors)
    np.save(os.path.join(output_dir, 'latent_labels.npy'), labels)
    
    results = {
        'reduced_vectors': reduced_vectors,
        'labels': labels,
        'method': method
    }
    
    return results


def run_training(config, train_loader, val_loader, experiment_dir):
    """
    Run the full training pipeline.
    
    Args:
        config: Configuration dictionary
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        experiment_dir: Directory to save results
        
    Returns:
        tuple: (trained_model, training_history)
    """
    # Set up directories
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Initialize model
    from .models import ACHNN
    model = ACHNN(config, num_classes=config['achnn_model']['num_classes'])
    model = model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0)
    )
    
    # Get training parameters
    num_epochs = config['training']['num_epochs']
    patience = config['training'].get('early_stopping_patience', 10)
    
    # Train the model
    logging.info(f'Starting training for {num_epochs} epochs with early stopping patience {patience}')
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=patience
    )
    
    # Save the training history
    history_path = os.path.join(experiment_dir, 'training_history.json')
    serializable_history = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_accuracy': history['val_accuracy']
    }
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    # Plot learning curves
    plot_path = os.path.join(experiment_dir, 'learning_curves.png')
    plot_learning_curves(
        history['train_loss'], 
        history['val_loss'], 
        history['val_accuracy'],
        plot_path
    )
    
    return model, history


def run_cv_fold(config, fold_idx, train_loader, val_loader, fold_dir, label_encoder):
    """
    Run a single cross-validation fold.
    
    Args:
        config: Configuration dictionary
        fold_idx: Index of the current fold
        train_loader: Training DataLoader for this fold
        val_loader: Validation DataLoader for this fold
        fold_dir: Directory to save results for this fold
        label_encoder: LabelEncoder object with class names
        
    Returns:
        dict: Fold results
    """
    os.makedirs(fold_dir, exist_ok=True)
    
    logging.info(f'Starting fold {fold_idx+1}')
    
    # Train the model
    model, history = run_training(config, train_loader, val_loader, fold_dir)
    
    # Get device
    device = next(model.parameters()).device
    
    # Evaluate the model
    criterion = nn.CrossEntropyLoss()
    class_names = label_encoder.classes_
    
    metrics = evaluate_model(
        model=model,
        test_loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )
    
    # Save confusion matrix
    cm_path = os.path.join(fold_dir, 'confusion_matrix.png')
    save_confusion_matrix_plot(metrics['confusion_matrix'], class_names, cm_path)
    
    # Analyze Hopfield attention patterns
    attention_dir = os.path.join(fold_dir, 'attention_analysis')
    attention_results = analyze_hopfield_attention(
        model=model,
        loader=val_loader,
        class_names=class_names,
        output_dir=attention_dir,
        device=device
    )
    
    # Visualize latent space
    latent_dir = os.path.join(fold_dir, 'latent_space')
    latent_results = visualize_latent_space(
        model=model,
        loader=val_loader,
        class_names=class_names,
        output_dir=latent_dir,
        device=device
    )
    
    # Save metrics
    metrics_path = os.path.join(fold_dir, 'metrics.json')
    serializable_metrics = {
        'accuracy': float(metrics['accuracy']),
        'f1_score': float(metrics['f1_score']),
        'test_loss': float(metrics['test_loss'])
    }
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    # Save predictions
    np.save(os.path.join(fold_dir, 'predictions.npy'), metrics['predictions'])
    np.save(os.path.join(fold_dir, 'targets.npy'), metrics['targets'])
    
    fold_results = {
        'metrics': metrics,
        'attention_results': attention_results,
        'latent_results': latent_results
    }
    
    return fold_results


def aggregate_cv_results(config, cv_results, aggregated_dir, label_encoder):
    """
    Aggregate results from all cross-validation folds.
    
    Args:
        config: Configuration dictionary
        cv_results: List of results from each fold
        aggregated_dir: Directory to save aggregated results
        label_encoder: LabelEncoder object with class names
        
    Returns:
        dict: Aggregated metrics
    """
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Aggregate metrics
    accuracies = [r['metrics']['accuracy'] for r in cv_results]
    f1_scores = [r['metrics']['f1_score'] for r in cv_results]
    test_losses = [r['metrics']['test_loss'] for r in cv_results]
    
    aggregated_metrics = {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'f1_score_mean': float(np.mean(f1_scores)),
        'f1_score_std': float(np.std(f1_scores)),
        'test_loss_mean': float(np.mean(test_losses)),
        'test_loss_std': float(np.std(test_losses))
    }
    
    # Save aggregated metrics
    metrics_path = os.path.join(aggregated_dir, 'aggregated_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    # Aggregate confusion matrices
    class_names = label_encoder.classes_
    cms = [r['metrics']['confusion_matrix'] for r in cv_results]
    aggregated_cm = np.sum(cms, axis=0)
    
    # Save aggregated confusion matrix
    cm_path = os.path.join(aggregated_dir, 'aggregated_confusion_matrix.png')
    save_confusion_matrix_plot(aggregated_cm, class_names, cm_path)
    
    # Aggregate Hopfield attention
    attention_matrices = [r['attention_results']['attention_matrix'] for r in cv_results if 'attention_matrix' in r['attention_results']]
    if attention_matrices:
        aggregated_attention = np.mean(np.stack(attention_matrices), axis=0)
        
        # Save aggregated attention matrix visualization
        attention_dir = os.path.join(aggregated_dir, 'attention_analysis')
        os.makedirs(attention_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(aggregated_attention, annot=True, fmt='.3f', cmap='viridis',
                    xticklabels=[f'Pattern {i+1}' for i in range(aggregated_attention.shape[1])],
                    yticklabels=class_names)
        plt.xlabel('Hopfield Stored Patterns')
        plt.ylabel('Class')
        plt.title('Aggregated Hopfield Attention by Class')
        plt.tight_layout()
        
        output_path = os.path.join(attention_dir, 'aggregated_hopfield_attention.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save the data
        np.save(os.path.join(attention_dir, 'aggregated_attention_matrix.npy'), aggregated_attention)
    
    logging.info("Aggregated cross-validation results:")
    logging.info(f"  Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
    logging.info(f"  F1 Score: {aggregated_metrics['f1_score_mean']:.4f} ± {aggregated_metrics['f1_score_std']:.4f}")
    logging.info(f"  Test Loss: {aggregated_metrics['test_loss_mean']:.4f} ± {aggregated_metrics['test_loss_std']:.4f}")
    
    return aggregated_metrics 