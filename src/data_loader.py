import os
import glob
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_subject_id_from_path # Assuming utils.py is in the same directory

class PainDataProcessor:
    """
    Processes timeseries and event data for a single subject run from ds000140 dataset.
    
    This class handles:
    1. Loading regional timeseries processed by the RPN-Signature pipeline
    2. Loading events.tsv files from the BIDS dataset
    3. Mapping event conditions to standardized labels
    4. Aligning labels to TRs considering HRF delay
    5. Optionally handling motion scrubbing information
    """
    def __init__(self, config, subject_id, run_id):
        self.config = config
        self.subject_id = subject_id
        self.run_id = run_id
        self.base_dir = config['paths']['base_dir']
        self.tr = config['data']['tr']
        self.hrf_delay_secs = config['data']['hrf_delay_secs']
        self.apply_scrubbing = config['data'].get('apply_scrubbing', False)
        self.fd_threshold = config['data'].get('fd_threshold', 0.5)
        
        # Construct file paths based on RPN-Signature output structure
        self.timeseries_path = os.path.join(
            self.base_dir,
            'regional-timeseries',
            f"{self.subject_id}_task-pain_run-{self.run_id}_timeseries.tsv"
        )
        
        # Events path is expected to be in BIDS format within the base_dir
        # You may need to adjust this based on actual file locations
        self.events_path = os.path.join(
            self.base_dir,
            self.subject_id,
            'func',
            f"{self.subject_id}_task-pain_run-{self.run_id}_events.tsv"
        )
        
        # Path to FD timeseries file for scrubbing
        if self.apply_scrubbing:
            self.fd_path = os.path.join(
                self.base_dir,
                'func_preproc',
                'mc_fd',
                f"{self.subject_id}_task-pain_run-{self.run_id}{config['data'].get('fd_file_suffix', '_FD.txt')}"
            )
        
        self.scaler = StandardScaler()
        
    def _load_timeseries(self):
        """
        Loads regional timeseries data from RPN-Signature output.
        
        Returns:
            numpy.ndarray: Matrix of shape (num_trs, num_regions) or None if error
        """
        try:
            ts_df = pd.read_csv(self.timeseries_path, sep='\t', header=None)
            logging.debug(f"Loaded timeseries from {self.timeseries_path}, shape: {ts_df.shape}")
            
            # Ensure correct number of regions (MIST122 atlas)
            expected_regions = self.config['data']['num_regions']
            if ts_df.shape[1] != expected_regions:
                logging.warning(f"Timeseries {self.timeseries_path} has {ts_df.shape[1]} regions, expected {expected_regions}.")
                
            return ts_df.values  # Return as numpy array
        except FileNotFoundError:
            logging.error(f"Timeseries file not found: {self.timeseries_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading timeseries {self.timeseries_path}: {e}")
            return None
    
    def _load_events(self):
        """
        Loads events data from ds000140 BIDS events.tsv file.
        
        Returns:
            pandas.DataFrame: Events data or None if error
        """
        try:
            events_df = pd.read_csv(self.events_path, sep='\t')
            logging.debug(f"Loaded events from {self.events_path}, columns: {events_df.columns.tolist()}")
            
            # Check for essential columns
            required_cols = ['onset', 'duration']
            if not all(col in events_df.columns for col in required_cols):
                logging.error(f"Missing required columns in events file: {self.events_path}")
                return None
                
            return events_df
        except FileNotFoundError:
            logging.error(f"Events file not found: {self.events_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading events {self.events_path}: {e}")
            return None
    
    def _load_fd_timeseries(self):
        """
        Loads framewise displacement timeseries for motion scrubbing.
        
        Returns:
            numpy.ndarray: Vector of FD values per TR or None if error/not enabled
        """
        if not self.apply_scrubbing:
            return None
            
        try:
            # FD files are typically simple text files with one value per line
            with open(self.fd_path, 'r') as f:
                fd_values = np.array([float(line.strip()) for line in f if line.strip()])
            
            logging.debug(f"Loaded FD timeseries from {self.fd_path}, length: {len(fd_values)}")
            return fd_values
        except FileNotFoundError:
            logging.warning(f"FD timeseries file not found: {self.fd_path}, scrubbing will be skipped")
            return None
        except Exception as e:
            logging.error(f"Error loading FD timeseries {self.fd_path}: {e}")
            return None
    
    def _map_condition_label(self, event_row):
        """
        Maps ds000140 event rows to standardized condition labels.
        
        ds000140 includes pain stimuli at varying intensities with different
        cognitive modulation instructions (passive experience, regulate UP, regulate DOWN).
        
        Args:
            event_row: pandas.Series with event information
            
        Returns:
            str: Standardized condition label
        """
        # Default label for non-experimental periods
        baseline_label = self.config['data'].get('baseline_label', 'baseline')
        
        # Extract relevant fields from the event
        # NOTE: These fields should be checked against actual ds000140 events.tsv columns
        try:
            # Check for required fields based on ds000140 format
            # Adjust these based on actual events.tsv column names
            trial_type = event_row.get('trial_type', None)
            
            # If this isn't a pain/task trial, return baseline
            if trial_type is None or 'pain' not in trial_type.lower():
                return baseline_label
                
            # Extract intensity information (may be in separate field or part of trial_type)
            intensity = None
            if 'intensity' in event_row:
                intensity = event_row['intensity']
            elif 'stim_intensity' in event_row:
                intensity = event_row['stim_intensity']
            else:
                # Try to extract from trial_type string
                if 'high' in trial_type.lower():
                    intensity = 'high'
                elif 'medium' in trial_type.lower() or 'med' in trial_type.lower():
                    intensity = 'medium'
                elif 'low' in trial_type.lower():
                    intensity = 'low'
                else:
                    intensity = 'unknown'
            
            # Extract modulation condition
            modulation = None
            if 'modulation_condition' in event_row:
                modulation = event_row['modulation_condition']
            elif 'cognitive_condition' in event_row:
                modulation = event_row['cognitive_condition']
            else:
                # Try to extract from trial_type or other fields
                if 'passive' in trial_type.lower():
                    modulation = 'passive_experience'
                elif any(kw in trial_type.lower() for kw in ['downreg', 'down_reg', 'down-reg']):
                    modulation = 'regulate_down'
                elif any(kw in trial_type.lower() for kw in ['upreg', 'up_reg', 'up-reg']):
                    modulation = 'regulate_up'
                else:
                    modulation = 'passive_experience'  # Default
            
            # Combine into standardized format
            condition_label = f"{intensity}_intensity_{modulation}"
            
            # Check if this condition is in the list of conditions to classify
            conditions_to_classify = self.config['data'].get('conditions_to_classify', [])
            if conditions_to_classify and condition_label not in conditions_to_classify:
                logging.debug(f"Condition '{condition_label}' not in conditions_to_classify, mapping to {baseline_label}")
                return baseline_label
                
            return condition_label
            
        except Exception as e:
            logging.warning(f"Error mapping condition label for event row {event_row}: {e}")
            return baseline_label

    def align_labels(self, timeseries_data, events_df, fd_values=None):
        """
        Aligns event labels to TRs considering HRF delay and optional scrubbing.
        
        Args:
            timeseries_data: numpy.ndarray of shape (num_trs, num_regions)
            events_df: pandas.DataFrame with event information
            fd_values: numpy.ndarray with FD values per TR for scrubbing (optional)
            
        Returns:
            numpy.ndarray: Array of string labels for each TR
        """
        num_trs = timeseries_data.shape[0]
        baseline_label = self.config['data'].get('baseline_label', 'baseline')
        aligned_labels = np.array([baseline_label] * num_trs, dtype=object)
        
        # Apply scrubbing if enabled and FD values are available
        scrubbed_trs = np.zeros(num_trs, dtype=bool)
        if self.apply_scrubbing and fd_values is not None:
            # Ensure fd_values has right length
            if len(fd_values) != num_trs:
                logging.warning(f"FD timeseries length ({len(fd_values)}) doesn't match timeseries length ({num_trs}). Truncating.")
                fd_values = fd_values[:num_trs] if len(fd_values) > num_trs else np.pad(fd_values, (0, num_trs - len(fd_values)))
                
            # Mark high-motion TRs
            scrubbed_trs = fd_values > self.fd_threshold
            logging.debug(f"Scrubbing {np.sum(scrubbed_trs)} of {num_trs} TRs with FD > {self.fd_threshold}")
        
        # Calculate HRF delay in TRs
        hrf_delay_trs = int(np.round(self.hrf_delay_secs / self.tr))
        logging.debug(f"Aligning labels with TR={self.tr}s, HRF delay={self.hrf_delay_secs}s ({hrf_delay_trs} TRs)")
        
        if events_df is None or events_df.empty:
            logging.warning(f"No events data for {self.subject_id} run {self.run_id}, returning all baseline labels")
            return aligned_labels, scrubbed_trs
        
        # Process each event
        for _, row in events_df.iterrows():
            # Convert onset and duration from seconds to TRs
            onset_tr = int(np.floor(row['onset'] / self.tr))
            duration_trs = int(np.ceil(row['duration'] / self.tr))
            
            # Apply HRF delay
            label_start_tr = onset_tr + hrf_delay_trs
            label_end_tr = label_start_tr + duration_trs
            
            # Ensure indices are within bounds
            label_start_tr = max(0, label_start_tr)
            label_end_tr = min(num_trs, label_end_tr)
            
            if label_start_tr < label_end_tr:
                condition_label = self._map_condition_label(row)
                if condition_label != baseline_label:
                    aligned_labels[label_start_tr:label_end_tr] = condition_label
                    logging.debug(f"Event '{condition_label}' mapped to TRs {label_start_tr}-{label_end_tr-1}")
        
        return aligned_labels, scrubbed_trs

    def normalize_timeseries(self, timeseries_data):
        """
        Normalizes timeseries data using StandardScaler.
        
        Args:
            timeseries_data: numpy.ndarray of shape (num_trs, num_regions)
            
        Returns:
            numpy.ndarray: Normalized timeseries
        """
        try:
            return self.scaler.fit_transform(timeseries_data)
        except ValueError as e:
            logging.error(f"Error normalizing timeseries for {self.subject_id} run {self.run_id}: {e}")
            raise

    def process(self):
        """
        Loads, aligns labels, handles scrubbing, and normalizes data for the run.
        
        Returns:
            tuple: (normalized_timeseries, aligned_labels, scrubbed_trs) or (None, None, None) if error
        """
        # Load timeseries data
        timeseries_data = self._load_timeseries()
        if timeseries_data is None:
            return None, None, None
            
        # Load events data
        events_df = self._load_events()
        
        # Load FD values for scrubbing if enabled
        fd_values = self._load_fd_timeseries() if self.apply_scrubbing else None
        
        # Align labels and mark scrubbed TRs
        aligned_labels, scrubbed_trs = self.align_labels(timeseries_data, events_df, fd_values)
        
        # Normalize timeseries
        try:
            normalized_ts = self.normalize_timeseries(timeseries_data)
        except Exception as e:
            logging.error(f"Normalization failed: {e}")
            return None, None, None
            
        return normalized_ts, aligned_labels, scrubbed_trs


def create_windows(timeseries, labels, scrubbed_trs=None, seq_len=10, step=1, baseline_label='baseline'):
    """
    Creates sliding windows from timeseries data, excluding baseline labels and optionally scrubbed TRs.
    
    Args:
        timeseries: numpy.ndarray of shape (num_trs, num_regions)
        labels: numpy.ndarray of string labels for each TR
        scrubbed_trs: numpy.ndarray of booleans marking high-motion TRs to exclude
        seq_len: Window length in TRs
        step: Step size for sliding window
        baseline_label: Label to exclude when creating windows
        
    Returns:
        tuple: (windows, window_labels, subject_ids)
    """
    num_trs, num_features = timeseries.shape
    window_data = []
    window_labels = []
    window_groups = []  # To store subject_id for GroupKFold
    
    # If scrubbed_trs is None, create an array of all False
    if scrubbed_trs is None:
        scrubbed_trs = np.zeros(num_trs, dtype=bool)
    
    # Create windows with slide_step
    for i in range(0, num_trs - seq_len + 1, step):
        end_idx = i + seq_len
        
        # Get label at the last TR of the window (adjusted for HRF delay already in align_labels)
        target_label = labels[end_idx - 1]
        
        # Only keep windows where:
        # 1. The target label is not baseline
        # 2. The window does not contain scrubbed TRs (if scrubbing is enabled)
        if target_label != baseline_label:
            if np.any(scrubbed_trs[i:end_idx]):
                # Skip windows containing scrubbed TRs
                continue
                
            window_ts = timeseries[i:end_idx, :]
            window_data.append(window_ts)
            window_labels.append(target_label)
            window_groups.append(1)  # Will be replaced with subject_id later
    
    if not window_data:
        logging.warning(f"No valid windows created with seq_len {seq_len}")
        return None, None, None
        
    return np.array(window_data), np.array(window_labels), np.array(window_groups)


def load_all_subject_data(config, included_subjects_file, results_dir):
    """
    Loads, processes, and windows data for all included subjects.
    
    Args:
        config: Configuration dictionary
        included_subjects_file: Path to file with included subject IDs
        results_dir: Directory to save label encoder
        
    Returns:
        tuple: (X_all, y_all, groups_all, label_encoder)
    """
    try:
        with open(included_subjects_file, 'r') as f:
            included_subjects = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(included_subjects)} subjects from {included_subjects_file}")
    except FileNotFoundError:
        logging.error(f"Included subjects file not found: {included_subjects_file}")
        raise
    
    all_window_data = []
    all_window_labels_str = []
    all_window_groups = []
    
    included_runs = config['data']['included_runs']
    seq_len = config['data']['seq_len']
    step = config['data'].get('window_step', 1)
    baseline_label = config['data'].get('baseline_label', 'baseline')
    
    for subject_id in included_subjects:
        logging.info(f"Processing subject: {subject_id}")
        subject_has_data = False
        
        for run_id in included_runs:
            logging.debug(f"Processing run: {run_id}")
            
            processor = PainDataProcessor(config, subject_id, run_id)
            normalized_ts, aligned_labels, scrubbed_trs = processor.process()
            
            if normalized_ts is not None and aligned_labels is not None:
                window_data, window_labels_str, window_groups = create_windows(
                    normalized_ts, aligned_labels, scrubbed_trs, 
                    seq_len=seq_len, step=step, baseline_label=baseline_label
                )
                
                if window_data is not None and len(window_data) > 0:
                    all_window_data.append(window_data)
                    all_window_labels_str.append(window_labels_str)
                    
                    # Replace placeholder group values with actual subject ID
                    groups = np.array([subject_id] * len(window_data))
                    all_window_groups.append(groups)
                    
                    subject_has_data = True
                    logging.debug(f"Added {len(window_data)} windows from {subject_id} run {run_id}")
                else:
                    logging.warning(f"No valid windows created for {subject_id} run {run_id}")
            else:
                logging.warning(f"Skipping {subject_id} run {run_id} due to processing errors")
        
        if not subject_has_data:
            logging.warning(f"No data loaded for subject {subject_id} across specified runs")
    
    if not all_window_data:
        logging.error("No valid window data could be loaded or processed for any subject/run")
        raise ValueError("Failed to load any valid data")
    
    # Concatenate data from all subjects/runs
    X_all = np.concatenate(all_window_data, axis=0).astype(np.float32)
    y_all_str = np.concatenate(all_window_labels_str, axis=0)
    groups_all = np.concatenate(all_window_groups, axis=0)
    
    logging.info(f"Total windows created: {X_all.shape[0]}")
    logging.info(f"Window shape: {X_all.shape[1:]}")
    logging.info(f"Unique labels found: {np.unique(y_all_str)}")
    
    # Fit and save LabelEncoder
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_all_str)
    
    logging.info("LabelEncoder fitted:")
    for i, class_name in enumerate(label_encoder.classes_):
        logging.info(f"  Class {i}: {class_name}")
    
    # Save label encoder
    encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
    try:
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        logging.info(f"LabelEncoder saved to {encoder_path}")
    except Exception as e:
        logging.error(f"Could not save LabelEncoder to {encoder_path}: {e}")
    
    return X_all, y_all, groups_all, label_encoder


class FMRIWindowDataset(Dataset):
    """PyTorch Dataset for fMRI windows"""
    def __init__(self, data, labels):
        """
        Args:
            data: numpy.ndarray of shape (num_windows, seq_len, num_regions)
            labels: numpy.ndarray of class indices
        """
        # Ensure data is float32 for features and long for labels
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # Input validation
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