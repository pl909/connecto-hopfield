#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis script for examining attention patterns and latent space representations
of a trained ACHNN model for ABIDE ASD vs TDC classification.
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
from torch.utils.data import DataLoader, Subset # Import Subset
import glob

# Add the parent directory to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models import ACHNN
# Updated import:
from src.data_loader import FMRIWindowDataset, load_all_subject_data 
# Updated import:
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

def load_best_model_from_cv(experiment_dir, config, device):
    """Loads the best model checkpoint from the cross-validation folds."""
    best_val_loss = float('inf')
    best_model_path = None
    best_fold = -1

    n_folds = config['training']['cv_folds']
    for fold in range(n_folds):
        fold_dir = os.path.join(experiment_dir, f"fold_{fold}")
        metrics_path = os.path.join(fold_dir, 'final_val_metrics.json')
        checkpoint_path = os.path.join(fold_dir, 'best_model.pt')

        if os.path.exists(metrics_path) and os.path.exists(checkpoint_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                # Use 'test_loss' key as saved by evaluate_model
                current_val_loss = metrics.get('test_loss') 
                if current_val_loss is not None and current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_model_path = checkpoint_path
                    best_fold = fold
            except Exception as e:
                logging.warning(f"Could not read metrics for fold {fold}: {e}")
        else:
            logging.warning(f"Metrics or checkpoint missing for fold {fold}")

    if best_model_path is None:
        raise FileNotFoundError(f"Could not find the best model checkpoint across folds in {experiment_dir}")

    logging.info(f"Loading best model from fold {best_fold} (Val Loss: {best_val_loss:.4f}) located at {best_model_path}")
    
    # Initialize model architecture based on config
    num_classes = config['achnn_model']['num_classes']
    model = ACHNN(config, num_classes=num_classes)
    
    # Load the state dict
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, best_fold


def load_data_for_analysis(config, experiment_dir):
    """Loads all data and label encoder for analysis purposes."""
    # Load label encoder first
    encoder_path = os.path.join(experiment_dir, 'label_encoder.pkl')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Load all data (needed for visualization across all subjects)
    phenotypic_file = os.path.join(config['paths']['base_dir'], config['paths']['phenotypic_file'])
    X_all, y_all, groups_all, _ = load_all_subject_data(
        config, phenotypic_file, experiment_dir # Pass pheno file path
    )
    
    # Create a full dataset and loader for analysis
    full_dataset = FMRIWindowDataset(X_all, y_all)
    analysis_loader = DataLoader(
        full_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False, # Important for consistent analysis
        num_workers=config['training'].get('num_workers', 2)
    )
    
    return analysis_loader, label_encoder, X_all, y_all, groups_all


def analyze_hopfield_patterns_abide(model, analysis_loader, class_names, output_dir, device):
    """
    Analyze Hopfield attention patterns comparing ASD vs TDC groups.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the imported analyze_hopfield_attention function
    # It already calculates average attention per class
    attention_results = analyze_hopfield_attention(
        model, analysis_loader, class_names, output_dir, device)
        
    # Optional: Further analysis specific to ASD vs TDC could be added here
    # e.g., statistical tests comparing attention patterns between the two groups.
    
    logging.info("Hopfield attention analysis comparing ASD vs TDC complete.")


def run_latent_space_analysis_abide(model, analysis_loader, class_names, output_dir, device):
    """
    Analyze latent space representations comparing ASD vs TDC groups.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run t-SNE visualization
    tsne_results = visualize_latent_space(
        model=model,
        loader=analysis_loader,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        method='tsne',
        perplexity=30
    )
    
    # Run PCA visualization
    pca_results = visualize_latent_space(
        model=model,
        loader=analysis_loader,
        class_names=class_names,
        output_dir=output_dir,
        device=device,
        method='pca',
        n_components=2
    )
    
    # Additional analysis: Compute distance between ASD and TDC centroids
    try:
        reduced_vectors = pca_results['reduced_vectors']
        labels = pca_results['labels'] # These are encoded labels (0, 1)
        
        centroids = {}
        for i, class_name in enumerate(class_names): # class_names should be ['TDC', 'ASD'] or similar
            mask = labels == i
            if np.any(mask):
                centroids[class_name] = np.mean(reduced_vectors[mask], axis=0)
        
        if len(centroids) == 2: # Ensure both classes are present
            class_list = list(centroids.keys())
            distance = np.linalg.norm(centroids[class_list[0]] - centroids[class_list[1]])
            logging.info(f"Euclidean distance between '{class_list[0]}' and '{class_list[1]}' centroids in PCA space: {distance:.4f}")
            
            # Save distance
            dist_data = {'distance': float(distance), 'class1': class_list[0], 'class2': class_list[1]}
            with open(os.path.join(output_dir, 'centroid_distance.json'), 'w') as f:
                json.dump(dist_data, f, indent=2)
                
    except Exception as e:
        logging.error(f"Error during centroid distance calculation: {e}")

def main():
    """Main function to run ACHNN analysis for ABIDE."""
    parser = argparse.ArgumentParser(description='Analyze trained ACHNN model for ABIDE ASD/TDC classification')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Directory containing trained model folds and configuration')
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
    
    # Add file handler for logging analysis steps
    log_file_handler = logging.FileHandler(os.path.join(args.output_dir, 'analysis.log'))
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(log_file_handler)
    
    # Set device
    device = get_device()
    
    try:
        # Load configuration from the experiment directory
        config_path = os.path.join(args.experiment_dir, 'config.yaml')
        if not os.path.exists(config_path):
             config_path = os.path.join(args.experiment_dir, 'config.yml')
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Configuration file not found in {args.experiment_dir}")
        config = load_config(config_path)

        # Load the best model from cross-validation
        model, best_fold_idx = load_best_model_from_cv(args.experiment_dir, config, device)
        
        # Load all data and label encoder for analysis
        analysis_loader, label_encoder, _, _, _ = load_data_for_analysis(config, args.experiment_dir)
        class_names = label_encoder.classes_.tolist() # Ensure it's a list
        # Map numerical class names (1, 2) to meaningful labels if needed
        class_map = {1: 'ASD', 2: 'TDC'} # Assuming DX_GROUP 1=ASD, 2=TDC
        display_class_names = [class_map.get(int(cls), str(cls)) for cls in class_names]

        logging.info(f"Starting analysis using best model from fold {best_fold_idx}")
        logging.info(f"Classes for analysis: {display_class_names}")
        
        # Create sub-directories for different analyses
        attention_dir = os.path.join(args.output_dir, 'attention_analysis')
        latent_dir = os.path.join(args.output_dir, 'latent_space')
        
        # Run Hopfield attention pattern analysis
        logging.info("Analyzing Hopfield attention patterns (ASD vs TDC)...")
        analyze_hopfield_patterns_abide(model, analysis_loader, display_class_names, attention_dir, device)
        
        # Run latent space analysis
        logging.info("Analyzing latent space representations (ASD vs TDC)...")
        run_latent_space_analysis_abide(model, analysis_loader, display_class_names, latent_dir, device)
        
        # Re-evaluate model performance on the full dataset (or a dedicated test set if available)
        # Note: Evaluating on the full dataset used for training/validation gives an optimistic performance estimate.
        # Ideally, you'd have a separate hold-out test set.
        logging.info("Evaluating best model performance on the full dataset (for reference)...")
        criterion = torch.nn.CrossEntropyLoss()
        metrics = evaluate_model(model, analysis_loader, criterion, device, display_class_names)
        
        # Save confusion matrix for the full dataset evaluation
        cm_path = os.path.join(args.output_dir, 'full_data_confusion_matrix.png')
        save_confusion_matrix_plot(metrics['confusion_matrix'], display_class_names, cm_path)
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, 'full_data_metrics.json')
        serializable_metrics = {
            'accuracy': float(metrics['accuracy']),
            'f1_score': float(metrics['f1_score']),
            'loss': float(metrics['test_loss']) # Renaming key for clarity
        }
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logging.info(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {e}", exc_info=True) # Log traceback
        raise

if __name__ == "__main__":
    main()