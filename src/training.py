import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
import os
import json
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from contextlib import nullcontext
import time
from tqdm import tqdm
import torch.nn.functional as F

# Import the initialize_model function from models.py
from models import initialize_model

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, metric='accuracy'):
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
            metric (str): Metric to monitor for early stopping ('accuracy' or 'loss')
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric = metric
        self.best_epoch = -1  # Initialize best_epoch

    def __call__(self, val_loss, val_accuracy, model, epoch):
        current_metric_value = val_accuracy if self.metric == 'accuracy' else -val_loss

        if self.best_score is None:
            self.best_score = current_metric_value
            self.save_checkpoint(val_loss, model, epoch)
        elif current_metric_value < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_metric_value
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """Saves model when the monitored metric improves."""
        if self.verbose:
            # Determine the correct message based on the metric
            if self.metric == 'accuracy':
                score_str = f"{self.best_score*100:.2f}%"
                self.trace_func(f'Validation accuracy increased ({score_str}). Saving model to {self.path} from epoch {epoch + 1}...')
            else: # metric == 'loss'
                self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} from epoch {epoch + 1}...')
        try:
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss  # Still track min loss for potential info
            self.best_epoch = epoch      # Store the epoch number
        except Exception as e:
            self.trace_func(f"Error saving model checkpoint to {self.path}: {e}")

# --- CORRECTED train_epoch function ---
def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, clip_value=None, scheduler=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader with training data
        criterion: Loss function
        optimizer: Optimizer to use
        device: Device to train on
        scaler: Optional GradScaler for mixed precision training
        clip_value: Value for gradient clipping
        scheduler: Optional learning rate scheduler to call after each batch
        
    Returns:
        dict: Training metrics including loss, accuracy, gradient norms, and learning rates
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []
    learning_rates = []
    batch_losses = []
    batch_accuracies = []
    
    print("Starting training phase...")
    print(f"Training on {len(train_loader.dataset)} samples...")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Use mixed precision if available
        with torch.amp.autocast('cuda') if device.type == 'cuda' else nullcontext():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with gradient scaling if mixed precision is enabled
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_value is not None:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                grad_norms.append(total_norm.item())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_value is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                grad_norms.append(total_norm.item())
            optimizer.step()
        
        # Update learning rate (if scheduler is used) and record it
        if scheduler is not None:
            scheduler.step()
            learning_rates.append(scheduler.get_last_lr()[0])
        else:
            learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        batch_size = target.size(0)
        correct += batch_correct
        total += batch_size
        total_loss += loss.item()
        
        # Record batch metrics
        batch_losses.append(loss.item())
        batch_accuracies.append(100 * batch_correct / batch_size)
        
        # Show progress at regular intervals
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            current_samples = total
            total_samples = len(train_loader.dataset)
            progress_pct = 100 * current_samples / total_samples
            batch_acc = 100 * batch_correct / batch_size
            running_acc = 100 * correct / total if total > 0 else 0
            current_lr = learning_rates[-1]
            grad_norm = grad_norms[-1] if grad_norms else 0.0
            
            print(f"Training: {batch_idx+1}/{len(train_loader)} batches ({progress_pct:.1f}%)")
            print(f"  - Batch Acc: {batch_acc:.2f}%, Running Acc: {running_acc:.2f}%")
            print(f"  - Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
            print(f"  - Learning Rate: {current_lr:.6f}")
        
        # Free memory
        del data, target, output, loss
        torch.cuda.empty_cache()
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = correct / total
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'grad_norms': grad_norms,
        'learning_rates': learning_rates,
        'batch_losses': batch_losses,
        'batch_accuracies': batch_accuracies
    }
    
    print("\nTraining phase complete")
    print(f"Epoch Summary:")
    print(f"  - Average Loss: {epoch_loss:.4f}")
    print(f"  - Average Accuracy: {100 * epoch_acc:.2f}%")
    print(f"  - Average Grad Norm: {np.mean(grad_norms):.4f}")
    print(f"  - Final Learning Rate: {learning_rates[-1]:.6f}")
    
    return metrics

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
    print("Starting validation phase...")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # Don't use model compilation here as it may cause issues with torch.dynamo
    # We'll use a simpler, more reliable approach
    
    print(f"Starting validation on {len(dataloader.dataset)} samples...")
    batch_size = dataloader.batch_size
    total_batches = len(dataloader)
    progress_interval = max(1, min(5, total_batches // 20))  # Report more frequently
    
    # Use a timeout monitor to detect stuck batches
    import time
    start_time = time.time()
    batch_start_time = time.time()
    max_time_per_batch = 60  # Set a 60-second timeout for any batch
    
    with torch.no_grad():
        print("Entering no_grad block for validation")
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Monitor time for each batch to detect hangs
            batch_elapsed = time.time() - batch_start_time
            if batch_elapsed > max_time_per_batch:
                print(f"WARNING: Batch {batch_idx} is taking longer than {max_time_per_batch}s ({batch_elapsed:.1f}s). "
                      f"This may indicate a processing issue.")
            
            # Reset batch timer
            batch_start_time = time.time()
            
            # Print frequent progress updates
            if batch_idx % progress_interval == 0:
                elapsed = time.time() - start_time
                estimated_remaining = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1) if batch_idx > 0 else 0
                print(f"Validation progress: {batch_idx}/{total_batches} batches ({100*batch_idx/total_batches:.1f}%)")
                print(f"  - Elapsed: {elapsed:.1f}s, Estimated remaining: {estimated_remaining:.1f}s")
            
            try:
                # Process validation in smaller chunks to avoid OOM
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Forward pass - disable gradient computation
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy (save these to CPU immediately to free GPU memory)
                _, preds = torch.max(outputs, 1)
                batch_correct = preds.eq(targets).sum().item()
                batch_total = targets.size(0)
                
                # Update metrics
                val_loss += loss.item() * batch_total
                correct += batch_correct
                total += batch_total
                
                # Aggressively clear memory
                del outputs, preds, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Show more detailed progress
                if (batch_idx + 1) % progress_interval == 0 or (batch_idx + 1) == total_batches:
                    current_samples = (batch_idx + 1) * batch_size
                    total_samples = len(dataloader.dataset)
                    progress_pct = 100 * current_samples / total_samples
                    running_acc = 100 * correct / total if total > 0 else 0
                    batch_acc = 100 * batch_correct / batch_total
                    print(f"Validation: {batch_idx+1}/{total_batches} batches ({progress_pct:.1f}%)")
                    print(f"  - Batch Acc: {batch_acc:.2f}%, Running Acc: {running_acc:.2f}%")
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"CUDA out of memory in validation batch {batch_idx+1}. Trying to recover...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    # Continue with the next batch
                    continue
                elif "expected scalar type" in str(e) or "expected dtype" in str(e):
                    print(f"Data type mismatch in validation batch {batch_idx+1}: {e}")
                    print(f"This may be related to mixed precision issues. Skipping batch.")
                    continue
                else:
                    print(f"Runtime error during validation batch {batch_idx+1}: {e}")
                    continue
            except Exception as e:
                print(f"Error during validation batch {batch_idx+1}: {e}")
                continue

            # Update inputs/targets
            del inputs, targets
    
    # Calculate final metrics
    val_loss = val_loss / total if total > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0
    
    # Report total validation time
    total_time = time.time() - start_time
    print(f"Validation complete in {total_time:.2f}s - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Return empty arrays for predictions since we're not storing them
    return val_loss, accuracy, np.array([]), np.array([])

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
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_predictions'].append(val_preds)
        history['val_targets'].append(val_targets)
        
        logging.info(f'Epoch {epoch+1}/{num_epochs}: '
                    f'Train Loss: {train_metrics["loss"]:.4f}, '
                    f'Val Loss: {val_loss:.4f}, '
                    f'Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        early_stopping(val_loss, val_accuracy, model, epoch)
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
    print("Starting final evaluation...")
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    test_loss = 0.0
    
    # Use larger batch size for evaluation if possible
    batch_size = test_loader.batch_size
    total_batches = len(test_loader)
    print(f"Evaluating {len(test_loader.dataset)} samples across {total_batches} batches")
    progress_interval = max(1, total_batches // 20) # Show progress every 5% or at least every batch
    
    with torch.no_grad():
        # IMPORTANT: No mixed precision context for evaluation - causes data type issues
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            try:
                # Print progress
                if batch_idx % progress_interval == 0:
                    print(f"Evaluating batch {batch_idx+1}/{total_batches}")
                
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Forward pass - expect only logits in eval mode
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Accumulate loss and accuracy
                batch_total = targets.size(0)
                batch_correct = preds.eq(targets).sum().item()
                
                test_loss += loss.item() * batch_total
                correct += batch_correct
                total += batch_total
                
                # For confusion matrix and detailed metrics, we need to store all predictions
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Free memory
                del outputs, preds, inputs, targets
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Show progress at regular intervals
                if (batch_idx + 1) % (progress_interval * 5) == 0 or (batch_idx + 1) == total_batches:
                    running_acc = 100 * correct / total if total > 0 else 0
                    print(f"Evaluation: {batch_idx+1}/{total_batches} batches, Running Acc: {running_acc:.2f}%, Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"Error during evaluation batch {batch_idx+1}: {e}")
                continue  # Skip problematic batches
    
    print("Computing final evaluation metrics")
    
    # Calculate metrics
    test_loss = test_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    if len(all_preds) == 0 or len(all_targets) == 0:
        print("Warning: No predictions or targets collected during evaluation")
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'test_loss': float('inf'),
            'confusion_matrix': np.zeros((2, 2))
        }
    
    # Calculate F1 score
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"Evaluation complete - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Return metrics
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'test_loss': float(test_loss),
        'confusion_matrix': cm
    }

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


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, grad_norms, learning_rates, output_path):
    """
    Plot learning curves including loss, accuracy, gradient norms, and learning rates.
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        train_acc: List of training accuracies
        val_acc: List of validation accuracies
        grad_norms: List of gradient norms
        learning_rates: List of learning rates
        output_path: Path to save the plot
    """
    epochs = range(1, len(train_loss) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, [100 * acc for acc in train_acc], 'b-', label='Training Accuracy')
    ax2.plot(epochs, [100 * acc for acc in val_acc], 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot gradient norms
    ax3.plot(epochs, grad_norms, 'g-', label='Gradient Norm')
    ax3.set_title('Gradient Norm vs. Epochs')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_yscale('log')  # Use log scale for gradient norms
    ax3.legend()
    ax3.grid(True)
    
    # Plot learning rates
    # Resample learning rates to match epoch count (take mean per epoch)
    lr_per_epoch = []
    steps_per_epoch = len(learning_rates) // len(epochs)
    for i in range(len(epochs)):
        start_idx = i * steps_per_epoch
        end_idx = (i + 1) * steps_per_epoch
        lr_per_epoch.append(np.mean(learning_rates[start_idx:end_idx]))
    
    ax4.plot(epochs, lr_per_epoch, 'm-', label='Learning Rate')
    ax4.set_title('Learning Rate vs. Epochs')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')  # Use log scale for learning rates
    ax4.legend()
    ax4.grid(True)
    
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
    
    # Initialize structures to hold model outputs
    feature_vectors_by_class = {class_name: [] for class_name in class_names}
    
    # We can't directly access attention patterns from the model as implemented
    # So we'll collect the latent representations instead
    print("Extracting latent representations for analysis...")
    latent_vectors, labels = model.extract_latent_representations(loader, device)
    
    # Organize by class
    for i, label in enumerate(labels):
        class_name = class_names[label]
        feature_vectors_by_class[class_name].append(latent_vectors[i])
    
    # Compute average feature vector per class
    avg_features_by_class = {}
    for class_name, vectors in feature_vectors_by_class.items():
        if vectors:
            avg_vector = np.mean(np.stack(vectors), axis=0)
            avg_features_by_class[class_name] = avg_vector
            
            # Log the average vector norm for this class
            vector_norm = np.linalg.norm(avg_vector)
            logging.info(f"Class '{class_name}' - Average feature vector norm: {vector_norm:.4f}")
    
    # Compute cosine similarity between class vectors
    similarity_matrix = np.zeros((len(class_names), len(class_names)))
    
    for i, class1 in enumerate(class_names):
        if class1 not in avg_features_by_class:
            continue
        vec1 = avg_features_by_class[class1]
        
        for j, class2 in enumerate(class_names):
            if class2 not in avg_features_by_class:
                continue
            vec2 = avg_features_by_class[class2]
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarity_matrix[i, j] = similarity
    
    # Plot similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.title('Cosine Similarity Between Class Latent Representations')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'class_similarity_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Latent representation analysis saved to {output_path}')
    
    # Create a PCA visualization of the class centroids
    if len(avg_features_by_class) > 1:
        # Convert to array
        feature_matrix = np.stack(list(avg_features_by_class.values()))
        class_labels = list(avg_features_by_class.keys())
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(feature_matrix)
        
        # Plot
        plt.figure(figsize=(12, 10))
        for i, class_name in enumerate(class_labels):
            plt.scatter(reduced_features[i, 0], reduced_features[i, 1], s=100, label=class_name)
            plt.text(reduced_features[i, 0], reduced_features[i, 1], class_name, fontsize=12)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title('PCA of Class Feature Centroids')
        plt.legend()
        plt.grid(True)
        
        output_path = os.path.join(output_dir, 'class_centroids_pca.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    results = {
        'avg_features_by_class': avg_features_by_class,
        'similarity_matrix': similarity_matrix
    }
    
    # Save the numeric data for further analysis
    np.save(os.path.join(output_dir, 'class_similarity_matrix.npy'), similarity_matrix)
    
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
        history['batch_accuracies'],
        history['batch_accuracies'],
        history['grad_norms'],
        history['learning_rates'],
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
    Aggregate metrics across CV folds.
    
    Args:
        config: Configuration dictionary
        cv_results: List of dictionaries containing the results for each fold
        aggregated_dir: Directory to save aggregated results
        label_encoder: Label encoder used for the class names
    """
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Get class names for plotting
    class_names = label_encoder.classes_
    
    # Aggregate metrics
    metrics = ['accuracy', 'f1_score', 'test_loss']
    agg_metrics = {}
    
    for metric in metrics:
        values = [r['metrics'][metric] for r in cv_results]
        agg_metrics[f'mean_{metric}'] = float(np.mean(values))
        agg_metrics[f'std_{metric}'] = float(np.std(values))
    
    # Save aggregated metrics
    metrics_path = os.path.join(aggregated_dir, 'aggregated_metrics.json')
    save_json_metrics(agg_metrics, metrics_path)
    logging.info(f"Saved aggregated metrics to {metrics_path}")
    
    # Log aggregated performance
    logging.info("=== Aggregated Performance (Mean ± Std across Folds) ===")
    logging.info(f"  Accuracy: {agg_metrics.get('mean_accuracy', 'N/A'):.4f} ± {agg_metrics.get('std_accuracy', 'N/A'):.4f}")
    logging.info(f"  F1 Score: {agg_metrics.get('mean_f1_score', 'N/A'):.4f} ± {agg_metrics.get('std_f1_score', 'N/A'):.4f}")
    logging.info(f"  Test Loss: {agg_metrics.get('mean_test_loss', 'N/A'):.4f} ± {agg_metrics.get('std_test_loss', 'N/A'):.4f}")
    
    # Aggregate confusion matrices
    confusion_matrices = [r['metrics']['confusion_matrix'] for r in cv_results]
    aggregated_cm = np.sum(confusion_matrices, axis=0)
    
    # Normalize aggregated confusion matrix by row (true labels)
    normalized_cm = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1)[:, np.newaxis]
    
    # Save confusion matrix plot
    cm_path = os.path.join(aggregated_dir, 'aggregated_confusion_matrix.png')
    save_confusion_matrix_plot(aggregated_cm, class_names, cm_path)
    
    # Aggregate learning curves
    train_losses = [np.array(r['train_log']['train_loss']) for r in cv_results]
    val_losses = [np.array(r['train_log']['val_loss']) for r in cv_results]
    train_accs = [np.array(r['train_log']['train_acc']) for r in cv_results]
    val_accs = [np.array(r['train_log']['val_acc']) for r in cv_results]
    
    # Find the minimum length among all arrays
    min_length = min([len(arr) for arr in train_losses + val_losses + train_accs + val_accs])
    
    # Truncate all arrays to the minimum length
    train_losses = [arr[:min_length] for arr in train_losses]
    val_losses = [arr[:min_length] for arr in val_losses]
    train_accs = [arr[:min_length] for arr in train_accs]
    val_accs = [arr[:min_length] for arr in val_accs]
    
    # Average across folds
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    mean_train_acc = np.mean(train_accs, axis=0)
    mean_val_acc = np.mean(val_accs, axis=0)
    
    # Calculate standard deviation for error bands
    std_train_loss = np.std(train_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    std_train_acc = np.std(train_accs, axis=0)
    std_val_acc = np.std(val_accs, axis=0)
    
    # Plot the learning curves with error bands
    epochs = np.arange(1, min_length + 1)
    
    # Plot losses
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, alpha=0.2, color='b')
    plt.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, alpha=0.2, color='r')
    plt.plot(epochs, mean_train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, mean_val_loss, 'r-', label='Validation Loss')
    plt.title('Loss vs. Epochs (Mean ± Std across Folds)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.fill_between(epochs, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha=0.2, color='b')
    plt.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, alpha=0.2, color='r')
    plt.plot(epochs, mean_train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, mean_val_acc, 'r-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs (Mean ± Std across Folds)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(aggregated_dir, 'aggregated_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aggregate Hopfield attention
    similarity_matrices = [r['attention_results']['similarity_matrix'] for r in cv_results if 'similarity_matrix' in r['attention_results']]
    if similarity_matrices:
        aggregated_similarity = np.mean(np.stack(similarity_matrices), axis=0)
        
        # Plot aggregated similarity matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(aggregated_similarity, annot=True, fmt='.3f', cmap='viridis',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.title('Average Similarity Between Class Latent Representations')
        plt.tight_layout()
        
        plt.savefig(os.path.join(aggregated_dir, 'aggregated_similarity_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save numeric data
        np.save(os.path.join(aggregated_dir, 'aggregated_similarity_matrix.npy'), aggregated_similarity)
    
    logging.info(f"Aggregated CV results saved to {aggregated_dir}")


def save_json_metrics(metrics_dict, output_path):
    """Saves evaluation metrics dictionary to a JSON file."""
    try:
        # Ensure values are JSON serializable (convert numpy types if needed)
        serializable_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (np.generic, np.ndarray)):
                serializable_metrics[key] = value.item() if value.size == 1 else value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                 serializable_metrics[key] = value
            else:
                 # Attempt conversion for other types, log warning if unknown
                 try:
                     serializable_metrics[key] = float(value) 
                 except (TypeError, ValueError):
                     logging.warning(f"Could not serialize metric '{key}' of type {type(value)}. Skipping.")
                     
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        logging.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {output_path}: {e}")

def save_training_log(fold_log_dict, output_csv_path):
    """Saves the epoch-wise training/validation logs to a CSV file."""
    try:
        # Ensure all lists have the same length for DataFrame creation
        # Find the length of the shortest list to avoid errors if training stopped early
        min_len = min(len(v) for v in fold_log_dict.values() if isinstance(v, list))
        trimmed_log = {k: v[:min_len] for k, v in fold_log_dict.items()}
        
        log_df = pd.DataFrame(trimmed_log)
        log_df.index.name = 'epoch'
        log_df.index = log_df.index + 1 # Start epoch count from 1
        log_df.to_csv(output_csv_path)
        logging.info(f"Training log saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving training log to {output_csv_path}: {e}")

def validate_epoch_regression(model, dataloader, criterion, device):
    """
    Validate the model on regression task for one epoch.
    
    Args:
        model: Model to validate
        dataloader: DataLoader containing validation data
        criterion: Loss function
        device: Device to use for validation
        
    Returns:
        dict: Dictionary containing validation metrics (val_loss, val_mse, val_mae)
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = len(dataloader)
    num_samples = 0
    
    print(f"Starting validation with {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"Validating batch {batch_idx}/{num_batches}")
                
            # Transfer data to device
            try:
                inputs = inputs.to(device)
                targets = targets.to(device).float()  # Ensure targets are float for regression
                
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze()  # Ensure outputs are same shape as targets
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Calculate MSE and MAE
                mse = torch.mean((outputs - targets) ** 2)
                mae = torch.mean(torch.abs(outputs - targets))
                
                # Update totals
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_mse += mse.item() * batch_size
                total_mae += mae.item() * batch_size
                num_samples += batch_size
                
                # Free up memory
                del inputs, targets, outputs
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
            except RuntimeError as e:
                # Handle CUDA out-of-memory error
                if "CUDA out of memory" in str(e):
                    print(f"WARNING: CUDA OOM during validation batch {batch_idx}. Skipping batch.")
                    torch.cuda.empty_cache() if device.type == 'cuda' else None
                    continue
                # Handle other errors like shape mismatch
                elif "shape" in str(e) or "size" in str(e):
                    print(f"WARNING: Shape/size error during validation batch {batch_idx}: {e}. Skipping batch.")
                    continue
                else:
                    raise e
    
    # Calculate average metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    avg_mse = total_mse / num_samples if num_samples > 0 else float('inf')
    avg_mae = total_mae / num_samples if num_samples > 0 else float('inf')
    
    # Add completion message
    print(f"Validation completed! Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")
    logging.info(f"Validation epoch complete - avg_loss: {avg_loss:.4f}, avg_mse: {avg_mse:.4f}, avg_mae: {avg_mae:.4f}")
    
    # Return metrics dictionary
    metrics = {
        'val_loss': avg_loss,
        'val_mse': avg_mse,
        'val_mae': avg_mae
    }
    
    return metrics

def evaluate_model_regression(model, dataloader, criterion, device):
    """
    Evaluate the model on regression task.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader containing test data
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    print("Starting model evaluation for regression...")
    model.eval()
    
    # Initialize lists to store predictions and targets
    predictions = []
    targets = []
    
    # Initialize variables for tracking
    correct = 0
    total = 0
    test_loss = 0.0
    num_batches = len(dataloader)
    
    # Use mixed precision if on CUDA
    if device.type == 'cuda':
        context_manager = torch.amp.autocast('cuda')
    else:
        context_manager = nullcontext()
    
    with torch.no_grad(), context_manager:
        for batch_idx, (inputs, batch_targets) in enumerate(dataloader):
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Evaluating batch {batch_idx}/{num_batches}")
                
            try:
                # Transfer data to device
                inputs = inputs.to(device)
                batch_targets = batch_targets.to(device).float()
                
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze()
                
                # Calculate loss
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item() * batch_targets.size(0)
                
                # Store predictions and targets
                predictions.extend(outputs.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
                
                # Update total
                total += batch_targets.size(0)
                
                # Free memory
                del inputs, batch_targets, outputs
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
            except RuntimeError as e:
                print(f"WARNING: Error during evaluation batch {batch_idx}: {e}. Skipping batch.")
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue
    
    # Calculate metrics
    if total > 0:
        test_loss /= total
        
        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate MSE and MAE
        if len(predictions) > 0 and len(targets) > 0:
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
        else:
            print("WARNING: No predictions or targets collected during evaluation.")
            mse = float('nan')
            mae = float('nan')
    else:
        print("WARNING: No samples were evaluated.")
        test_loss = float('nan')
        mse = float('nan')
        mae = float('nan')
        predictions = []
        targets = []
    
    # Return metrics and predictions
    result = {
        'test_loss': test_loss,
        'test_mse': mse,
        'test_mae': mae,
        'predictions': predictions,
        'targets': targets
    }
    
    return result

def plot_regression_history(history, save_dir):
    """
    Plot training and validation metrics for regression tasks.
    
    Args:
        history: Dictionary containing training metrics
        save_dir: Directory to save plots
    """
    # Create the plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_history.png'), dpi=150)
    plt.close()
    
    # Plot validation MSE and MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Validation Error Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'error_metrics.png'), dpi=150)
    plt.close()
    
    # Save history data as CSV
    history_df = pd.DataFrame(history)
    history_df.index.name = 'epoch'
    history_df.index = history_df.index + 1  # Start epochs from 1
    history_df.to_csv(os.path.join(plots_dir, 'training_history.csv'))
    
    logging.info(f"Regression plots saved to {plots_dir}")

# Add nullcontext for Python < 3.7
class nullcontext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Add missing utility functions for optimizer and scheduler
def configure_optimizer(model, config):
    """
    Configure optimizer based on the configuration.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    # Get optimizer parameters
    lr = config.get('training', {}).get('learning_rate', 0.001)
    weight_decay = config.get('training', {}).get('weight_decay', 0.0)
    optimizer_name = config.get('training', {}).get('optimizer', 'adam').lower()
    
    # Create optimizer based on name
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('training', {}).get('momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logging.warning(f"Unknown optimizer '{optimizer_name}'. Using Adam.")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    logging.info(f"Using {optimizer.__class__.__name__} optimizer with learning rate {lr} and weight decay {weight_decay}")
    return optimizer

def configure_scheduler(optimizer, dataloader, config):
    """
    Configure learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        dataloader: DataLoader containing training data (needed for some schedulers)
        config: Configuration dictionary
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler or None: Configured scheduler or None if not enabled
    """
    scheduler_name = config.get('training', {}).get('scheduler', None)
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower()
    num_epochs = config.get('training', {}).get('num_epochs', 100)
    
    if scheduler_name == 'cosine':
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif scheduler_name == 'step':
        # Step decay scheduler
        step_size = config.get('training', {}).get('lr_step_size', 30)
        gamma = config.get('training', {}).get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_name == 'plateau':
        # Reduce on plateau scheduler
        patience = config.get('training', {}).get('lr_patience', 10)
        factor = config.get('training', {}).get('lr_factor', 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    elif scheduler_name == 'linear':
        # Linear decay scheduler using LambdaLR
        lambda_fn = lambda epoch: 1.0 - (epoch / float(num_epochs))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    elif scheduler_name == 'onecycle':
        # One cycle scheduler
        steps_per_epoch = len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.get('training', {}).get('learning_rate', 0.001),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
    else:
        logging.warning(f"Unknown scheduler '{scheduler_name}'. Not using any scheduler.")
        return None
    
    logging.info(f"Using {scheduler.__class__.__name__} scheduler")
    return scheduler

def train_model_regression(train_loader, val_loader, test_loader, config, fold_dir, device, fold_idx=0):
    """
    Train a model for regression tasks and evaluate its performance.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        config: Configuration dictionary
        fold_dir: Directory to save fold-specific results
        device: Device to use for training
        fold_idx: Index of the current fold
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logging.info(f"Initializing model for regression task (fold {fold_idx})")
    
    # Model initialization
    model = initialize_model(config, task_type='regression')
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = configure_optimizer(model, config)
    scheduler = configure_scheduler(optimizer, train_loader, config)
    criterion = torch.nn.MSELoss()
    
    # Initialize variables for early stopping
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Initialize history for tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': []
    }
    
    # Training loop
    logging.info(f"Starting training for {config['training']['num_epochs']} epochs")
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        train_loss = train_epoch_regression(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler
        )
        
        # Validation phase
        val_metrics = validate_epoch_regression(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        val_loss = val_metrics['val_loss']
        val_mse = val_metrics['val_mse']
        val_mae = val_metrics['val_mae']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        
        # Print progress
        logging.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val MSE: {val_mse:.4f}, "
            f"Val MAE: {val_mae:.4f}"
        )
        
        # Check for improvement
        if val_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            
            # Save best model
            best_model_path = os.path.join(fold_dir, f"best_model.pt")
            torch.save({
                'model_state_dict': best_model_state,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_mse': val_mse,
                'val_mae': val_mae,
                'config': config
            }, best_model_path)
            logging.info(f"Best model saved to {best_model_path}")
        else:
            early_stopping_counter += 1
            logging.info(f"Validation loss did not improve for {early_stopping_counter} epochs")
            
            if early_stopping_counter >= config['training']['early_stopping_patience']:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training history
    plot_regression_history(history, fold_dir)
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    logging.info("Evaluating best model on test set")
    test_metrics = evaluate_model_regression(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Save test predictions
    test_predictions_path = os.path.join(fold_dir, "test_predictions.csv")
    pd.DataFrame({
        'true': test_metrics['targets'],
        'predicted': test_metrics['predictions']
    }).to_csv(test_predictions_path, index=False)
    
    # Calculate R-squared
    if len(test_metrics['targets']) > 1:
        from sklearn.metrics import r2_score
        r2 = r2_score(test_metrics['targets'], test_metrics['predictions'])
        test_metrics['test_r2'] = r2
        logging.info(f"Test R²: {r2:.4f}")
    
    # Save test metrics
    test_results = {
        'test_mse': test_metrics['test_mse'],
        'test_mae': test_metrics['test_mae'],
        'test_r2': test_metrics.get('test_r2', float('nan'))
    }
    
    test_results_path = os.path.join(fold_dir, "test_results.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    logging.info(f"Test results saved to {test_results_path}")
    
    return test_results


def train_epoch_regression(model, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch on regression task.
    
    Args:
        model: Model to train
        dataloader: DataLoader containing training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to use for training
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        float: Average training loss for this epoch
    """
    print("=== Starting training epoch ===")
    logging.info("Beginning training epoch")
    
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Use mixed precision if on CUDA
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Transfer data to device
        inputs = inputs.to(device)
        targets = targets.to(device).float()  # Ensure targets are float for regression
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        # Update learning rate if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            logging.info(f"Train Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item():.4f}")
    
    # Step scheduler if not OneCycleLR
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau needs validation loss
            pass  # Will be stepped after validation
        else:
            scheduler.step()
    
    # Return average loss
    avg_loss = total_loss / num_batches
    
    print(f"=== Training epoch completed! Average loss: {avg_loss:.4f} ===")
    logging.info(f"Training epoch completed with average loss: {avg_loss:.4f}")
    
    return avg_loss