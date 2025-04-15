#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to load ABIDE data and run SVM classification with memory-efficient processing.
"""

import os
import sys
import logging
import yaml
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import psutil
import gc
from datetime import datetime
from scipy import sparse
import resource
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to the path to import project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_loader import load_all_subject_data
from src.utils import setup_logging, load_config

def set_memory_limit(limit_gb):
    """Set memory limit in GB"""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_limit = limit_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    logging.info(f"Set memory limit to {limit_gb}GB")

def validate_paths(config):
    """Validate that all required paths exist"""
    paths_to_check = [
        ('Base directory', config['paths']['base_dir']),
        ('Phenotypic file', os.path.join(config['paths']['base_dir'], config['paths']['phenotypic_file'])),
        ('Timeseries directory', config['paths']['regional_timeseries_dir']),
    ]
    
    for path_name, path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")
        logging.info(f"Validated {path_name}: {path}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"Detailed memory info:")
    logging.info(f"  RSS (Resident Set Size): {mem_info.rss / 1024 / 1024:.2f} MB")
    logging.info(f"  VMS (Virtual Memory Size): {mem_info.vms / 1024 / 1024:.2f} MB")
    if hasattr(mem_info, 'shared'):
        logging.info(f"  Shared: {mem_info.shared / 1024 / 1024:.2f} MB")
    if hasattr(mem_info, 'data'):
        logging.info(f"  Data: {mem_info.data / 1024 / 1024:.2f} MB")
    
    # Get system memory info
    sys_mem = psutil.virtual_memory()
    logging.info(f"System memory:")
    logging.info(f"  Total: {sys_mem.total / 1024 / 1024:.2f} MB")
    logging.info(f"  Available: {sys_mem.available / 1024 / 1024:.2f} MB")
    logging.info(f"  Used: {sys_mem.used / 1024 / 1024:.2f} MB")
    logging.info(f"  Free: {sys_mem.free / 1024 / 1024:.2f} MB")
    logging.info(f"  Percent used: {sys_mem.percent}%")
    
    return mem_info.rss / 1024 / 1024  # Convert to MB

def log_array_info(name, arr):
    """Log information about a numpy array"""
    if arr is None:
        logging.info(f"{name} is None")
        return
    
    logging.info(f"{name} info:")
    logging.info(f"  Shape: {arr.shape}")
    logging.info(f"  Size: {arr.size}")
    logging.info(f"  Memory usage: {arr.nbytes / 1024 / 1024:.2f} MB")
    logging.info(f"  dtype: {arr.dtype}")
    if isinstance(arr, np.ndarray):
        logging.info(f"  Non-zero elements: {np.count_nonzero(arr)}")
        if arr.size > 0:
            logging.info(f"  Min: {np.min(arr)}, Max: {np.max(arr)}")
            if np.isnan(arr).any():
                logging.info(f"  Contains NaN values: {np.isnan(arr).sum()}")
            if not np.isfinite(arr).all():
                logging.info(f"  Contains infinite values: {(~np.isfinite(arr)).sum()}")

def log_memory_status(stage):
    """Log memory usage at a given processing stage"""
    mem_usage = get_memory_usage()
    logging.info(f"Memory usage at {stage}: {mem_usage:.2f} MB")
    return mem_usage

def process_data_in_chunks(X_all, chunk_size=50):
    """
    Process data in chunks and convert to sparse matrix format.
    Returns a sparse matrix of flattened features.
    """
    try:
        n_samples = X_all.shape[0]
        n_features = X_all.shape[1] * X_all.shape[2]
        logging.info(f"Processing {n_samples} samples with {n_features} features")
        log_array_info("Input data", X_all)
        
        # Initialize list to store sparse matrices
        sparse_chunks = []
        total_memory = 0
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = X_all[i:end_idx]
            
            # Log chunk info
            log_array_info(f"Chunk {i}-{end_idx}", chunk)
            
            # Validate chunk data
            if np.isnan(chunk).any():
                logging.warning(f"NaN values found in chunk {i}-{end_idx}")
                chunk = np.nan_to_num(chunk, 0)
            
            if not np.isfinite(chunk).all():
                logging.warning(f"Infinite values found in chunk {i}-{end_idx}")
                chunk = np.clip(chunk, -1e6, 1e6)
            
            # Flatten chunk and convert to sparse matrix
            flattened = chunk.reshape(end_idx - i, -1)
            log_array_info(f"Flattened chunk {i}-{end_idx}", flattened)
            
            sparse_chunk = sparse.csr_matrix(flattened)
            total_memory += sparse_chunk.data.nbytes + sparse_chunk.indptr.nbytes + sparse_chunk.indices.nbytes
            logging.info(f"Sparse chunk memory: {total_memory / 1024 / 1024:.2f} MB")
            
            sparse_chunks.append(sparse_chunk)
            
            # Log progress and memory
            logging.info(f"Processed chunk {i}-{end_idx}/{n_samples}")
            log_memory_status(f"after processing chunk {i}-{end_idx}")
            
            # Clear memory aggressively
            del chunk, flattened
            gc.collect()
            
            # Force garbage collection
            gc.collect()
            
            # Check available memory
            mem = psutil.virtual_memory()
            if mem.available < 1024 * 1024 * 1024:  # Less than 1GB available
                logging.warning(f"Low memory warning! Only {mem.available / 1024 / 1024:.2f} MB available")
        
        # Vertically stack all sparse chunks in smaller batches
        logging.info("Stacking sparse chunks...")
        batch_size = 10  # Process 10 chunks at a time
        final_chunks = []
        
        for i in range(0, len(sparse_chunks), batch_size):
            batch = sparse_chunks[i:i+batch_size]
            stacked = sparse.vstack(batch)
            final_chunks.append(stacked)
            
            # Log batch info
            logging.info(f"Stacked batch {i}-{i+len(batch)}")
            logging.info(f"Batch memory: {(stacked.data.nbytes + stacked.indptr.nbytes + stacked.indices.nbytes) / 1024 / 1024:.2f} MB")
            
            del batch
            gc.collect()
        
        X_sparse = sparse.vstack(final_chunks)
        logging.info("Sparse matrix created successfully")
        logging.info(f"Final sparse matrix memory: {(X_sparse.data.nbytes + X_sparse.indptr.nbytes + X_sparse.indices.nbytes) / 1024 / 1024:.2f} MB")
        return X_sparse
        
    except Exception as e:
        logging.error(f"Error in process_data_in_chunks: {str(e)}", exc_info=True)
        raise

def process_subject_data(subject_file, window_size=30, step_size=1):
    """Process a single subject's data with memory-efficient windowing"""
    try:
        # Load data in chunks
        data_chunks = []
        for chunk in pd.read_csv(subject_file, chunksize=1000, header=None, delimiter='\t'):
            data_chunks.append(chunk.values)
        data = np.concatenate(data_chunks)
        
        # Validate data dimensions
        if len(data.shape) != 2:
            logging.error(f"Invalid data shape for {subject_file}: {data.shape}, expected 2 dimensions")
            return None
            
        # Data should be (timepoints, ROIs)
        if data.shape[1] != 200:  # CC200 atlas has 200 ROIs
            logging.error(f"Invalid number of ROIs in {subject_file}: {data.shape[1]}, expected 200")
            return None
            
        # Calculate number of windows
        n_timepoints = data.shape[0]
        n_windows = ((n_timepoints - window_size) // step_size) + 1
        
        if n_windows <= 0:
            logging.error(f"Invalid number of windows for {subject_file}: {n_windows}")
            return None
        
        # Process windows in batches
        windows = []
        batch_size = 100  # Process 100 windows at a time
        processed_windows = 0
        
        for i in range(0, n_windows, batch_size):
            end_idx = min(i + batch_size, n_windows)
            batch_windows = []
            
            for w in range(i, end_idx):
                start = w * step_size
                end = start + window_size
                if end > n_timepoints:
                    logging.error(f"Invalid window indices for {subject_file}: {start}:{end}")
                    continue
                window = data[start:end]  # This will be (window_size, ROIs)
                if window.shape != (window_size, 200):
                    logging.error(f"Invalid window shape for {subject_file}: {window.shape}, expected ({window_size}, 200)")
                    continue
                batch_windows.append(window)
                processed_windows += 1
            
            if batch_windows:  # Only extend if we have valid windows
                windows.extend(batch_windows)
            
            # Log memory usage and progress
            if i % 1000 == 0:
                logging.info(f"Processed {processed_windows}/{n_windows} windows")
                log_memory_status(f"window processing {processed_windows}/{n_windows}")
        
        if not windows:
            logging.error(f"No valid windows created for {subject_file}")
            return None
            
        windows_array = np.array(windows)  # Shape should be (n_windows, window_size, ROIs)
        # Validate final shape
        expected_shape = (len(windows), window_size, 200)
        if windows_array.shape != expected_shape:
            logging.error(f"Invalid windows shape for {subject_file}: {windows_array.shape}, expected {expected_shape}")
            return None
            
        logging.info(f"Successfully processed {subject_file}: {windows_array.shape} with {processed_windows} windows")
        return windows_array
    
    except Exception as e:
        logging.error(f"Error processing subject data {subject_file}: {str(e)}", exc_info=True)
        return None

def run_svm_on_windows(X_all, y_all):
    """
    Runs memory-efficient SVM classification on windowed fMRI data.
    """
    try:
        logging.info(f"Starting SVM processing at {datetime.now()}")
        logging.info(f"Original data shape: {X_all.shape}")
        log_memory_status("before processing")
        
        # Validate input data
        if not isinstance(X_all, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(X_all)}")
        
        if np.isnan(X_all).any():
            logging.warning("NaN values found in input data")
            X_all = np.nan_to_num(X_all, 0)
        
        # Convert to sparse matrix in chunks with progress tracking
        logging.info("Converting data to sparse format in chunks...")
        total_chunks = (X_all.shape[0] + 99) // 100  # Using chunk_size=100
        X_sparse = None
        processed_chunks = 0
        
        for i in range(0, X_all.shape[0], 100):
            end_idx = min(i + 100, X_all.shape[0])
            chunk = X_all[i:end_idx]
            
            # Log chunk info
            logging.info(f"Processing chunk {processed_chunks + 1}/{total_chunks}: {chunk.shape}")
            
            # Validate chunk data
            if np.isnan(chunk).any():
                chunk = np.nan_to_num(chunk, 0)
            
            if not np.isfinite(chunk).all():
                chunk = np.clip(chunk, -1e6, 1e6)
            
            # Flatten chunk and convert to sparse matrix
            flattened = chunk.reshape(end_idx - i, -1)
            sparse_chunk = sparse.csr_matrix(flattened)
            
            if X_sparse is None:
                X_sparse = sparse_chunk
            else:
                X_sparse = sparse.vstack([X_sparse, sparse_chunk])
            
            processed_chunks += 1
            logging.info(f"Processed {processed_chunks}/{total_chunks} chunks")
            log_memory_status(f"after chunk {processed_chunks}")
            
            # Clear memory
            del chunk, flattened, sparse_chunk
            gc.collect()
        
        # Free original data
        del X_all
        gc.collect()
        log_memory_status("after conversion to sparse")
        
        logging.info(f"Sparse matrix shape: {X_sparse.shape}")
        logging.info(f"Sparse matrix density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}")
        
        # Split data with progress logging
        logging.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_sparse, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Free memory
        del X_sparse
        gc.collect()
        log_memory_status("after split")
        
        # Train SVM with progress logging
        logging.info("Training LinearSVC...")
        clf = LinearSVC(
            dual="auto",
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            tol=1e-4,
            verbose=1
        )
        
        logging.info("Starting SVM training...")
        clf.fit(X_train, y_train)
        logging.info("SVM training completed")
        log_memory_status("after training")
        
        # Predict in smaller chunks with progress tracking
        logging.info("Predicting on test set in chunks...")
        chunk_size = 500
        y_pred = []
        total_test_chunks = (X_test.shape[0] + chunk_size - 1) // chunk_size
        
        for i in range(0, X_test.shape[0], chunk_size):
            end_idx = min(i + chunk_size, X_test.shape[0])
            X_test_chunk = X_test[i:end_idx]
            y_pred_chunk = clf.predict(X_test_chunk)
            y_pred.extend(y_pred_chunk)
            chunk_num = (i // chunk_size) + 1
            logging.info(f"Predicted chunk {chunk_num}/{total_test_chunks} ({end_idx}/{X_test.shape[0]} samples)")
            gc.collect()
        
        y_pred = np.array(y_pred)
        logging.info("Prediction completed")
        
        # Calculate and save metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Control", "ASD"])
        
        logging.info(f"SVM Accuracy: {accuracy}")
        logging.info("Classification Report:\n" + report)
        
        # Save results
        results_dir = os.path.join(project_root, 'results', 'svm_abide')
        os.makedirs(results_dir, exist_ok=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'timestamp': datetime.now().isoformat(),
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'total_windows': len(y_all),
            'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        results_file = os.path.join(results_dir, 'svm_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"SVM Classification Results\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            f.write(f"Data Information:\n")
            f.write(f"Total windows: {results['total_windows']}\n")
            f.write(f"Training samples: {results['training_samples']}\n")
            f.write(f"Test samples: {results['test_samples']}\n\n")
            f.write(f"Accuracy: {results['accuracy']}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
        
        logging.info(f"Results saved to {results_file}")
        log_memory_status("final")
        
        return results
        
    except Exception as e:
        logging.error(f"Error in run_svm_on_windows: {str(e)}", exc_info=True)
        raise

def load_all_subject_data(config, phenotypic_file, results_dir):
    """Load all subject data with memory-efficient processing"""
    try:
        # Load phenotypic data
        pheno_df = pd.read_csv(phenotypic_file)
        logging.info(f"Loaded phenotypic data for {len(pheno_df)} subjects from {phenotypic_file}")
        
        # Filter by site
        site_mask = pheno_df['SITE_ID'].isin(config['qc']['included_sites'])
        pheno_df = pheno_df[site_mask]
        logging.info(f"Filtered by site: {len(pheno_df)} subjects remaining from included sites.")
        
        # Filter by valid FILE_ID
        file_mask = pheno_df['FILE_ID'].notna()
        pheno_df = pheno_df[file_mask]
        logging.info(f"Filtered by valid FILE_ID: {len(pheno_df)} subjects remaining.")
        
        # Filter by valid DX_GROUP
        dx_mask = pheno_df['DX_GROUP'].isin([1, 2])
        pheno_df = pheno_df[dx_mask]
        logging.info(f"Filtered by valid DX_GROUP (1 or 2): {len(pheno_df)} subjects remaining.")
        
        # Initialize lists for data
        X_chunks = []  # Store chunks directly instead of all_windows
        labels = []
        groups = []
        failed_subjects = []
        processed_subjects = 0
        
        # Process each subject
        for idx, row in pheno_df.iterrows():
            subject_file = os.path.join(
                config['paths']['regional_timeseries_dir'],
                f"{row['FILE_ID']}_rois_cc200.1D"
            )
            
            try:
                # Process subject data
                windows = process_subject_data(
                    subject_file,
                    window_size=config['data']['seq_len'],
                    step_size=config['data']['window_step']
                )
                
                if windows is not None:
                    # Validate windows shape
                    if len(windows.shape) != 3:
                        logging.error(f"Invalid windows shape for subject {row['FILE_ID']}: {windows.shape}")
                        failed_subjects.append(row['FILE_ID'])
                        continue
                        
                    n_windows = len(windows)
                    X_chunks.append(windows)  # Store directly in X_chunks
                    labels.extend([row['DX_GROUP']] * n_windows)
                    groups.extend([row['FILE_ID']] * n_windows)
                    processed_subjects += 1
                    
                    # Log progress
                    if (idx + 1) % 10 == 0:
                        logging.info(f"Processed {idx + 1}/{len(pheno_df)} subjects")
                        log_memory_status(f"subject processing {idx + 1}")
                        logging.info(f"Current data shapes:")
                        for i, chunk in enumerate(X_chunks):
                            logging.info(f"  Chunk {i}: {chunk.shape}")
                else:
                    failed_subjects.append(row['FILE_ID'])
                
                # Clear memory
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing subject {row['FILE_ID']}: {str(e)}")
                failed_subjects.append(row['FILE_ID'])
                continue
        
        logging.info(f"Successfully processed {processed_subjects} subjects, "
                    f"failed to process {len(failed_subjects)} subjects.")
        
        if failed_subjects:
            logging.warning(f"Failed subjects: {failed_subjects}")
        
        # Validate all chunks have the same dimensions before concatenating
        if not X_chunks:
            raise ValueError("No valid data chunks created")
            
        expected_shape = X_chunks[0].shape[1:]  # Shape without the number of windows
        invalid_chunks = []
        
        for i, chunk in enumerate(X_chunks):
            if chunk.shape[1:] != expected_shape:
                logging.error(f"Chunk {i} has invalid shape: {chunk.shape}, expected (..., {expected_shape})")
                invalid_chunks.append(i)
        
        if invalid_chunks:
            raise ValueError(f"Found {len(invalid_chunks)} chunks with invalid shapes")
        
        # Concatenate valid chunks
        logging.info("Concatenating data chunks...")
        X_all = np.concatenate(X_chunks, axis=0)
        y_all = np.array(labels)
        groups_all = np.array(groups)
        
        # Validate final arrays
        if len(X_all) != len(y_all) or len(X_all) != len(groups_all):
            raise ValueError(f"Mismatched lengths: X_all ({len(X_all)}), y_all ({len(y_all)}), groups ({len(groups_all)})")
        
        logging.info(f"Final data shape: {X_all.shape}")
        
        # Create and save label encoder
        label_encoder = LabelEncoder()
        y_all = label_encoder.fit_transform(y_all)
        
        with open(os.path.join(results_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        return X_all, y_all, groups_all, label_encoder
        
    except Exception as e:
        logging.error(f"Error in load_all_subject_data: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Setup logging with timestamp
        log_path = os.path.join(project_root, 'logs', f'svm_abide_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        setup_logging(log_path)
        
        logging.info(f"Starting ABIDE SVM analysis at {datetime.now()}")
        log_memory_status("initial")
        
        # Set memory limit to 48GB (increased from 32GB)
        set_memory_limit(48)
        logging.info("Memory limit increased to 48GB")
        
        # Create config dictionary with the correct paths
        config = {
            'paths': {
                'base_dir': project_root,
                'phenotypic_file': 'Phenotypic_V1_0b_preprocessed1.csv',
                'regional_timeseries_dir': '/usr/project/xtmp/results/abide_timeseries/ccs_filt_noglobal_rois_cc200',
                'results_base_dir': os.path.join(project_root, 'results')
            },
            'qc': {
                'included_sites': [
                    'LEUVEN_1', 'LEUVEN_2', 'MAX_MUN', 'NYU', 'OHSU', 'PITT',
                    'SDSU', 'STANFORD', 'TRINITY', 'UCLA_1', 'UCLA_2', 'UM_1',
                    'UM_2', 'USM', 'YALE'
                ]
            },
            'data': {
                'num_regions': 200,  # CC200 atlas
                'tr': 2.0,
                'seq_len': 30,
                'window_step': 1
            }
        }
        
        # Validate paths before proceeding
        validate_paths(config)
        
        # Load data
        phenotypic_file = os.path.join(config['paths']['base_dir'], config['paths']['phenotypic_file'])
        results_dir = os.path.join(config['paths']['results_base_dir'], 'svm_abide')
        os.makedirs(results_dir, exist_ok=True)
        
        logging.info("Loading and preparing ABIDE data...")
        logging.info(f"Phenotypic file: {phenotypic_file}")
        logging.info(f"Timeseries directory: {config['paths']['regional_timeseries_dir']}")
        log_memory_status("before loading data")
        
        X_all, y_all, groups_all, label_encoder = load_all_subject_data(
            config, phenotypic_file, results_dir
        )
        
        # Validate loaded data
        if X_all is None or y_all is None:
            raise ValueError("Data loading failed: X_all or y_all is None")
        
        if len(X_all) != len(y_all):
            raise ValueError(f"Mismatched lengths: X_all ({len(X_all)}) != y_all ({len(y_all)})")
        
        log_memory_status("after loading data")
        logging.info(f"Data loaded: {X_all.shape[0]} total windows across {len(np.unique(groups_all))} subjects")
        logging.info(f"Class labels (encoded): {np.unique(y_all)}")
        logging.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        # Run SVM analysis
        logging.info("Running SVM classification...")
        results = run_svm_on_windows(X_all, y_all)
        
        # Save results
        results_file = os.path.join(results_dir, 'svm_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"SVM Classification Results\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            f.write(f"Data Information:\n")
            f.write(f"Total windows: {results['total_windows']}\n")
            f.write(f"Training samples: {results['training_samples']}\n")
            f.write(f"Test samples: {results['test_samples']}\n\n")
            f.write(f"Accuracy: {results['accuracy']}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
        
        logging.info(f"Results saved to {results_file}")
        log_memory_status("final")
        
    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}", exc_info=True)
        log_memory_status("at error")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Data validation error: {str(e)}", exc_info=True)
        log_memory_status("at error")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        log_memory_status("at error")
        sys.exit(1)

if __name__ == "__main__":
    main() 