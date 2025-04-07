#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis script for examining attention patterns and latent space representations
of a trained ACHNN (Attentive Connectome-based Hopfield Network) model.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import yaml
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import glob

# Add the parent directory to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models import ACHNN
from src.data_loader import FMRIWindowDataset, load_all_subject_data
from src.training import (
    analyze_hopfield_attention, visualize_latent_space, evaluate_model, save_confusion_matrix_plot
)

def get_device():
    """Determine and return the appropriate torch device (CUDA/CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("CUDA not available, using CPU")
    return device

def load_model_and_data(experiment_dir, device):
    """
    Load a trained model, configuration, and validation data from specified experiment directory.
    
    Args:
        experiment_dir: Directory containing the experiment results
        device: Device to load the model to
        
    Returns:
        tuple: (model, config, val_loader, label_encoder)
    """
    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(experiment_dir, 'config.yml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found in {experiment_dir}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load label encoder
    encoder_path = os.path.join(experiment_dir, 'label_encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Determine which model checkpoint to load
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    best_model_path = None
    
    # First check for checkpoints in the main directory
    if os.path.exists(checkpoint_dir):
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if not os.path.exists(best_model_path):
            # Try to find any .pt file in the checkpoint directory
            pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if pt_files:
                best_model_path = os.path.join(checkpoint_dir, pt_files[0])
    
    # If no checkpoint was found, look in fold directories
    if best_model_path is None:
        # Look for fold directories
        fold_pattern = os.path.join(experiment_dir, "fold_*")
        fold_dirs = sorted(glob.glob(fold_pattern))
        
        if fold_dirs:
            # Try the first fold as default
            fold_dir = fold_dirs[0]
            logging.info(f"No checkpoint found in main directory, trying fold directory: {fold_dir}")
            best_model_path = os.path.join(fold_dir, 'best_model.pt')
            
            if not os.path.exists(best_model_path):
                # Try any fold with a model file
                for fold_dir in fold_dirs:
                    candidate_path = os.path.join(fold_dir, 'best_model.pt')
                    if os.path.exists(candidate_path):
                        best_model_path = candidate_path
                        logging.info(f"Found model checkpoint in: {fold_dir}")
                        break
    
    if best_model_path is None or not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No model checkpoint found in {experiment_dir} or its fold directories")
    
    # Initialize model with the same configuration
    model = ACHNN(config, num_classes=len(label_encoder.classes_))
    
    # Load model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {best_model_path}")
    
    # Load validation data
    included_subjects_file = config.get('paths', {}).get('included_subjects')
    if not included_subjects_file or not os.path.exists(included_subjects_file):
        logging.warning(f"Included subjects file not found, will attempt to load data from experiment directory")
        
        # Try to load validation data if saved in the experiment directory
        val_data_path = os.path.join(experiment_dir, 'validation_data.npz')
        if os.path.exists(val_data_path):
            data = np.load(val_data_path)
            X_val = data['X_val']
            y_val = data['y_val']
        else:
            raise FileNotFoundError(f"No validation data found for analysis.")
    else:
        # Load the data for all subjects and create validation set
        # In a real scenario, you would only load the validation subjects
        # This is a simplification for analysis purposes
        X, y, groups, _ = load_all_subject_data(
            config, included_subjects_file, experiment_dir
        )
        
        # Use first 20% as validation for analysis (simplified approach)
        # In a real scenario, this would be determined by the cross-validation split
        split_idx = int(0.2 * len(X))
        X_val, y_val = X[:split_idx], y[:split_idx]
    
    # Create validation dataset and loader
    val_dataset = FMRIWindowDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get('training', {}).get('batch_size', 32),
        shuffle=False  # Don't shuffle for evaluation
    )
    
    logging.info(f"Loaded validation data with {len(val_dataset)} samples")
    
    return model, config, val_loader, label_encoder

def analyze_hopfield_patterns(model, val_loader, class_names, output_dir, device):
    """
    Analyze Hopfield attention patterns for each class.
    
    Args:
        model: Trained ACHNN model
        val_loader: DataLoader for validation data
        class_names: List of class names
        output_dir: Directory to save analysis results
        device: Device to run model on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the analyze_hopfield_attention function from training.py
    attention_results = analyze_hopfield_attention(model, val_loader, class_names, output_dir, device)
    
    # Additional analysis: Compare attention patterns between different conditions
    # For example, compare pain intensities or different modulation conditions
    
    try:
        # Parse class names to extract conditions
        attention_matrix = attention_results['attention_matrix']
        
        # Group classes by intensity (high, medium, low)
        intensity_groups = {}
        for i, class_name in enumerate(class_names):
            if 'high_intensity' in class_name:
                group = 'high_intensity'
            elif 'medium_intensity' in class_name:
                group = 'medium_intensity'
            elif 'low_intensity' in class_name:
                group = 'low_intensity'
            else:
                continue
                
            if group not in intensity_groups:
                intensity_groups[group] = []
            intensity_groups[group].append(i)
        
        # If we have at least two intensity groups, create a comparison
        if len(intensity_groups) >= 2:
            # Average attention by intensity
            intensity_attention = {}
            for intensity, indices in intensity_groups.items():
                if indices:
                    intensity_attention[intensity] = np.mean(attention_matrix[indices], axis=0)
            
            # Plot intensity comparison
            plt.figure(figsize=(12, 6))
            for intensity, attn in intensity_attention.items():
                plt.plot(np.arange(len(attn)), attn, 'o-', label=intensity)
            plt.xlabel('Hopfield Pattern Index')
            plt.ylabel('Average Attention')
            plt.title('Hopfield Attention by Pain Intensity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            intensity_path = os.path.join(output_dir, 'attention_by_intensity.png')
            plt.savefig(intensity_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved intensity-based analysis to {intensity_path}")
            
        # Similarly, group by modulation condition (passive, regulate_up, regulate_down)
        modulation_groups = {}
        for i, class_name in enumerate(class_names):
            if 'passive_experience' in class_name:
                group = 'passive_experience'
            elif 'regulate_up' in class_name:
                group = 'regulate_up'
            elif 'regulate_down' in class_name:
                group = 'regulate_down'
            else:
                continue
                
            if group not in modulation_groups:
                modulation_groups[group] = []
            modulation_groups[group].append(i)
        
        # If we have at least two modulation groups, create a comparison
        if len(modulation_groups) >= 2:
            # Average attention by modulation
            modulation_attention = {}
            for modulation, indices in modulation_groups.items():
                if indices:
                    modulation_attention[modulation] = np.mean(attention_matrix[indices], axis=0)
            
            # Plot modulation comparison
            plt.figure(figsize=(12, 6))
            for modulation, attn in modulation_attention.items():
                plt.plot(np.arange(len(attn)), attn, 'o-', label=modulation)
            plt.xlabel('Hopfield Pattern Index')
            plt.ylabel('Average Attention')
            plt.title('Hopfield Attention by Cognitive Modulation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            modulation_path = os.path.join(output_dir, 'attention_by_modulation.png')
            plt.savefig(modulation_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Saved modulation-based analysis to {modulation_path}")
    
    except Exception as e:
        logging.error(f"Error during additional attention pattern analysis: {e}")

def run_latent_space_analysis(model, val_loader, class_names, output_dir, device):
    """
    Analyze latent space representations for different pain conditions.
    
    Args:
        model: Trained ACHNN model
        val_loader: DataLoader for validation data
        class_names: List of class names
        output_dir: Directory to save analysis results
        device: Device to run model on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run t-SNE visualization
    tsne_results = visualize_latent_space(
        model=model,
        loader=val_loader,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        method='tsne',
        perplexity=30
    )
    
    # Run PCA visualization
    pca_results = visualize_latent_space(
        model=model,
        loader=val_loader,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        method='pca',
        n_components=2
    )
    
    # Additional latent space analysis: Compute distances between class centroids
    try:
        # Use the PCA results as they preserve distances better than t-SNE
        reduced_vectors = pca_results['reduced_vectors']
        labels = pca_results['labels']
        
        # Compute centroids for each class
        centroids = {}
        for i, class_name in enumerate(class_names):
            mask = labels == i
            if np.any(mask):
                centroids[class_name] = np.mean(reduced_vectors[mask], axis=0)
        
        # Compute pairwise distances between centroids
        class_names_present = list(centroids.keys())
        n_classes = len(class_names_present)
        distance_matrix = np.zeros((n_classes, n_classes))
        
        for i, class1 in enumerate(class_names_present):
            for j, class2 in enumerate(class_names_present):
                distance_matrix[i, j] = np.linalg.norm(centroids[class1] - centroids[class2])
        
        # Plot distance matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=class_names_present,
                    yticklabels=class_names_present)
        plt.xlabel('Class')
        plt.ylabel('Class')
        plt.title('Euclidean Distance Between Class Centroids in Latent Space')
        plt.tight_layout()
        
        distance_path = os.path.join(output_dir, 'centroid_distances.png')
        plt.savefig(distance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved centroid distance analysis to {distance_path}")
        
        # Save the distance matrix as CSV for further analysis
        dist_df = pd.DataFrame(distance_matrix, index=class_names_present, columns=class_names_present)
        dist_df.to_csv(os.path.join(output_dir, 'centroid_distances.csv'))
        
    except Exception as e:
        logging.error(f"Error during additional latent space analysis: {e}")

def main():
    """Main function to run ACHNN analysis."""
    parser = argparse.ArgumentParser(description='Analyze trained ACHNN model')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Directory containing trained model and configuration')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save analysis results (default: experiment_dir/analysis)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'analysis')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_device()
    
    try:
        # Load model and data
        model, config, val_loader, label_encoder = load_model_and_data(args.experiment_dir, device)
        class_names = label_encoder.classes_
        
        logging.info(f"Starting analysis of model from {args.experiment_dir}")
        logging.info(f"Model has {model.hopfield_num_stored_patterns} Hopfield stored patterns")
        logging.info(f"Classes: {', '.join(class_names)}")
        
        # Create sub-directories for different analyses
        attention_dir = os.path.join(args.output_dir, 'attention_analysis')
        latent_dir = os.path.join(args.output_dir, 'latent_space')
        
        # Run Hopfield attention pattern analysis
        logging.info("Analyzing Hopfield attention patterns...")
        analyze_hopfield_patterns(model, val_loader, class_names, attention_dir, device)
        
        # Run latent space analysis
        logging.info("Analyzing latent space representations...")
        run_latent_space_analysis(model, val_loader, class_names, latent_dir, device)
        
        # Evaluate model performance
        logging.info("Evaluating model performance...")
        criterion = torch.nn.CrossEntropyLoss()
        metrics = evaluate_model(model, val_loader, criterion, device, class_names)
        
        # Save confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        save_confusion_matrix_plot(metrics['confusion_matrix'], class_names, cm_path)
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        serializable_metrics = {
            'accuracy': float(metrics['accuracy']),
            'f1_score': float(metrics['f1_score']),
            'test_loss': float(metrics['test_loss'])
        }
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logging.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 