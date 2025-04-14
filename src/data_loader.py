import os
import glob
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import re # Import regex

# Assuming utils.py is in the same directory or accessible via sys.path
from .utils import get_subject_id_from_path 

class AbideTimeseriesProcessor:
    """
    Loads and processes a single resting-state fMRI timeseries file from ABIDE Preprocessed.
    Handles normalization.
    """
    def __init__(self, config, file_id):
        self.config = config
        self.file_id = file_id # e.g., Pitt_0050003
        
        # Construct timeseries path from config
        self.timeseries_dir = config['paths']['regional_timeseries_dir']
        
        # Get the derivative name directly from config or infer if needed
        # It's better to rely on the config if possible for clarity
        self.derivative_name = config['data'].get('derivative_name', 'rois_cc200') # Default to cc200 if not in config
        
        # --- CORRECTED FILENAME CONSTRUCTION ---
        self.timeseries_filename = f"{self.file_id}_{self.derivative_name}.1D"
        self.timeseries_path = os.path.join(
            self.timeseries_dir,
            self.timeseries_filename
        )
        # --- END CORRECTION ---

        self.scaler = StandardScaler()
        self.expected_regions = config['data']['num_regions']

    def _load_timeseries(self):
        """
        Loads regional timeseries data from a .1D file.
        These files often have whitespace delimiters and may or may not have a header.
        We'll try reading assuming whitespace delimiter and check shape.
        
        Returns:
            numpy.ndarray: Matrix of shape (num_trs, num_regions) or None if error
        """
        if not os.path.exists(self.timeseries_path):
            # Log the *exact* path being checked
            logging.error(f"Timeseries file not found: {self.timeseries_path}") 
            return None
            
        try:
            # Try reading with whitespace delimiter, assuming no header initially
            # Use sep='\s+' which is more robust for whitespace
                        # Try reading with whitespace delimiter, skipping comment lines starting with '#'
            # Also explicitly state no header row for data columns

                        # Use sep=r'\s+' (raw string) for whitespace and comment='#' to ignore header/comment lines
            ts_data = pd.read_csv(
                self.timeseries_path, 
                sep=r'\s+',     # <--- Corrected line (added 'r' prefix)
                header=None,    
                comment='#'     
            ).values
            
            # Basic check if shape seems plausible (at least some TRs, expected regions)
            if ts_data.ndim != 2 or ts_data.shape[1] != self.expected_regions:
                 logging.warning(f"Read data shape {ts_data.shape} from {self.timeseries_path} "
                                f"does not match expected regions ({self.expected_regions}). "
                                f"Check file format or config.")
                 return None 

            logging.debug(f"Loaded timeseries from {self.timeseries_path}, shape: {ts_data.shape}")
            return ts_data.astype(np.float32) # Ensure float type
            
        except pd.errors.EmptyDataError:
             logging.error(f"Timeseries file is empty: {self.timeseries_path}")
             return None
        except Exception as e:
            logging.error(f"Error loading/parsing timeseries {self.timeseries_path}: {e}")
            return None

    def normalize_timeseries(self, timeseries_data):
        """
        Normalizes timeseries data using StandardScaler.
        
        Args:
            timeseries_data: numpy.ndarray of shape (num_trs, num_regions)
            
        Returns:
            numpy.ndarray: Normalized timeseries
        """
        if timeseries_data is None or timeseries_data.shape[0] == 0:
             logging.warning(f"Cannot normalize empty timeseries for {self.file_id}")
             return None
        try:
            # Fit and transform
            return self.scaler.fit_transform(timeseries_data)
        except ValueError as e:
            logging.error(f"Error normalizing timeseries for {self.file_id}: {e}")
            if "Input contains NaN" in str(e) or "contains non-finite values" in str(e):
                 logging.error("Data contains NaN or Inf.")
                 return None
            elif "Numerical issues" in str(e) or "variance is zero" in str(e):
                 logging.warning("Zero variance detected in some features. Returning original data.")
                 return timeseries_data 
            else:
                 raise

    def process(self):
        """
        Loads and normalizes timeseries data for the subject/scan.
        
        Returns:
            numpy.ndarray: Normalized timeseries or None if error
        """
        timeseries_data = self._load_timeseries()
        if timeseries_data is None:
            return None
            
        normalized_ts = self.normalize_timeseries(timeseries_data)
        return normalized_ts

# --- create_resting_state_windows function remains the same ---
def create_resting_state_windows(timeseries, subject_id, seq_len=30, step=1):
    """
    Creates sliding windows from resting-state timeseries data.
    
    Args:
        timeseries: numpy.ndarray of shape (num_trs, num_regions)
        subject_id: Identifier for the subject (used for grouping in CV)
        seq_len: Window length in TRs
        step: Step size for sliding window
        
    Returns:
        tuple: (windows, window_subject_ids) or (None, None) if no windows
    """
    if timeseries is None or timeseries.shape[0] < seq_len:
        logging.warning(f"Timeseries for subject {subject_id} is too short ({timeseries.shape[0] if timeseries is not None else 'None'} TRs) for window length {seq_len}.")
        return None, None

    num_trs, num_features = timeseries.shape
    window_data = []
    window_subject_ids = [] 
    
    for i in range(0, num_trs - seq_len + 1, step):
        end_idx = i + seq_len
        window_ts = timeseries[i:end_idx, :]
        
        # Basic check for NaNs/Infs in the window
        if not np.isfinite(window_ts).all():
            logging.warning(f"Skipping window starting at TR {i} for subject {subject_id} due to non-finite values.")
            continue
            
        window_data.append(window_ts)
        window_subject_ids.append(subject_id) 
    
    if not window_data:
        logging.warning(f"No valid windows created for subject {subject_id} with seq_len {seq_len}")
        return None, None
        
    return np.array(window_data), np.array(window_subject_ids)


# --- load_all_subject_data function remains largely the same, just ensure it uses the corrected processor ---
def load_all_subject_data(config, phenotypic_file, results_dir):
    """
    Loads, processes, and windows resting-state data for all included subjects from ABIDE.
    
    Args:
        config: Configuration dictionary
        phenotypic_file: Path to the main ABIDE phenotypic CSV file
        results_dir: Directory to save label encoder
        
    Returns:
        tuple: (X_all, y_all, groups_all, label_encoder)
    """
    try:
        pheno_df = pd.read_csv(phenotypic_file)
        logging.info(f"Loaded phenotypic data for {len(pheno_df)} subjects from {phenotypic_file}")
    except FileNotFoundError:
        logging.error(f"Phenotypic file not found: {phenotypic_file}")
        raise
    except Exception as e:
        logging.error(f"Error reading phenotypic file {phenotypic_file}: {e}")
        raise

    # --- Filter Subjects based on Config ---
    included_sites = config['qc'].get('included_sites', None)
    if included_sites:
        pheno_df = pheno_df[pheno_df['SITE_ID'].isin(included_sites)].copy()
        logging.info(f"Filtered by site: {len(pheno_df)} subjects remaining from included sites.")
    
    pheno_df = pheno_df[pheno_df['FILE_ID'] != 'no_filename'].copy()
    pheno_df.dropna(subset=['FILE_ID'], inplace=True)
    logging.info(f"Filtered by valid FILE_ID: {len(pheno_df)} subjects remaining.")

    pheno_df = pheno_df[pheno_df['DX_GROUP'].isin([1, 2])].copy()
    logging.info(f"Filtered by valid DX_GROUP (1 or 2): {len(pheno_df)} subjects remaining.")
    
    # --- Process Each Included Subject ---
    all_window_data = []
    all_window_labels = []
    all_window_groups = []
    
    seq_len = config['data']['seq_len']
    step = config['data'].get('window_step', 1)
    
    subjects_processed = 0
    subjects_failed = 0

    for index, row in pheno_df.iterrows():
        # Use SUB_ID for grouping, FILE_ID for finding the file
        subject_id_num = int(row['SUB_ID'])
        subject_id_str = f"sub-{subject_id_num:07d}" # Use this consistent format for grouping
        file_id = row['FILE_ID']
        dx_group = int(row['DX_GROUP']) # 1 for ASD, 2 for Control
        
        logging.debug(f"Processing subject: {subject_id_str} (FILE_ID: {file_id})")
        
        # Use the corrected processor
        processor = AbideTimeseriesProcessor(config, file_id) 
        normalized_ts = processor.process()
        
        if normalized_ts is not None:
            window_data, window_subject_ids = create_resting_state_windows(
                normalized_ts, 
                subject_id_str, # Pass the consistent subject ID string
                seq_len=seq_len, 
                step=step
            )
            
            if window_data is not None and len(window_data) > 0:
                all_window_data.append(window_data)
                labels_for_subject = np.array([dx_group] * len(window_data))
                all_window_labels.append(labels_for_subject)
                all_window_groups.append(window_subject_ids) 
                
                logging.debug(f"Added {len(window_data)} windows for {subject_id_str}")
                subjects_processed += 1
            else:
                logging.warning(f"No valid windows generated for {subject_id_str} (FILE_ID: {file_id})")
                subjects_failed += 1
        else:
            logging.warning(f"Timeseries processing failed for {subject_id_str} (FILE_ID: {file_id})")
            subjects_failed += 1

    logging.info(f"Successfully processed {subjects_processed} subjects, failed to process {subjects_failed} subjects.")

    if not all_window_data:
        logging.error("No valid window data could be loaded or processed for any subject.")
        raise ValueError("Failed to load any valid data")
    
    # Concatenate data from all subjects
    X_all = np.concatenate(all_window_data, axis=0).astype(np.float32)
    y_all_raw_labels = np.concatenate(all_window_labels, axis=0) 
    groups_all = np.concatenate(all_window_groups, axis=0) 
    
    logging.info(f"Total windows created: {X_all.shape[0]}")
    logging.info(f"Window shape: {X_all.shape[1:]}")
    logging.info(f"Unique DX_GROUP labels found: {np.unique(y_all_raw_labels)}")
    
    # Fit LabelEncoder (maps 1 -> 0, 2 -> 1, or similar)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all_raw_labels) 
    
    logging.info("LabelEncoder mapping:")
    for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        logging.info(f"  Original DX_GROUP {original_label} -> Encoded {encoded_label}")
    
    # Save label encoder
    encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        logging.info(f"LabelEncoder saved to {encoder_path}")
    except Exception as e:
        logging.error(f"Could not save LabelEncoder to {encoder_path}: {e}")
    
    return X_all, y_all, groups_all, label_encoder


# --- FMRIWindowDataset class remains the same ---
class FMRIWindowDataset(Dataset):
    """PyTorch Dataset for fMRI windows"""
    def __init__(self, data, labels):
        """
        Args:
            data: numpy.ndarray of shape (num_windows, seq_len, num_regions)
            labels: numpy.ndarray of class indices (e.g., 0 or 1)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if self.data.ndim != 3:
            raise ValueError(f"Input data must be 3D (num_windows, seq_len, num_features), got shape {self.data.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"Input labels must be 1D (num_windows,), got shape {self.labels.shape}")
        if self.data.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Mismatch between number of data samples ({self.data.shape[0]}) and labels ({self.labels.shape[0]})")
        
        logging.debug(f"FMRIWindowDataset created with {len(self.data)} windows of shape {self.data.shape[1:]}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]