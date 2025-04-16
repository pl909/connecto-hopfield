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
from typing import List, Dict, Tuple, Optional, Union

# Fix the relative import
try:
    from utils import get_subject_id_from_path
except ImportError:
    try:
        from .utils import get_subject_id_from_path
    except ImportError:
        # Define a simple fallback function if utils cannot be imported
        def get_subject_id_from_path(file_path):
            """Extract subject ID from a file path"""
            filename = os.path.basename(file_path)
            # Try to extract ID from patterns like "site_ID" or just "ID"
            parts = re.split(r'[_-]', filename)
            if len(parts) > 1:
                # If format is like "site_ID_suffix", return ID
                return parts[1]
            else:
                # If just a simple filename, return without extension
                return os.path.splitext(filename)[0]

# Add load_abide_timeseries function to bridge the gap
def load_abide_timeseries(base_dir: str, phenotypic_file: str, timeseries_dir: str, 
                         included_sites: List[str], num_regions: int, tr: float,
                         window_length: int, window_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ABIDE timeseries data and prepare it for processing.
    
    Args:
        base_dir (str): Base directory for the project
        phenotypic_file (str): Path to the phenotypic data file
        timeseries_dir (str): Directory containing timeseries files
        included_sites (List[str]): List of site IDs to include
        num_regions (int): Number of brain regions
        tr (float): TR value in seconds
        window_length (int): Length of sliding window
        window_step (int): Step size for sliding window
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Time series data and labels
    """
    # Create a config dictionary compatible with load_all_subject_data
    config = {
        'paths': {
            'base_dir': base_dir,
            'regional_timeseries_dir': timeseries_dir
        },
        'qc': {
            'included_sites': included_sites
        },
        'data': {
            'num_regions': num_regions,
            'tr': tr,
            'seq_len': window_length,
            'window_step': window_step
        }
    }
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the data using the existing function
    pheno_path = os.path.join(base_dir, phenotypic_file)
    X_all, y_all, groups_all, _ = load_all_subject_data(config, pheno_path, results_dir)
    
    return X_all, y_all


class AbideTimeseriesProcessor:
    """
    Loads and processes a single resting-state fMRI timeseries file from ABIDE Preprocessed.
    Handles normalization.
    """
    def __init__(self, config, file_id):
        self.config = config
        self.file_id = file_id # e.g., Pitt_0050003
        
        # Construct timeseries path from config
        self.timeseries_dir = config.get('paths', {}).get('regional_timeseries_dir', '')
        
        # Get the derivative name directly from config or infer if needed
        # It's better to rely on the config if possible for clarity
        self.derivative_name = config.get('data', {}).get('derivative_name', 'rois_cc200') # Default to cc200 if not in config
        
        # Determine timeseries file format and path
        # Try different combinations to find the file
        self.find_timeseries_file()
        
        self.scaler = StandardScaler()
        self.expected_regions = config.get('data', {}).get('num_regions', 116)
        
        logging.debug(f"Initialized processor for subject {file_id}, looking for file: {self.timeseries_path}")

    def find_timeseries_file(self):
        """Find the correct timeseries file by trying different formats"""
        # Extract just the numeric part of the subject ID if it's a pure number
        numeric_id = self.file_id
        if str(numeric_id).isdigit():
            numeric_id = str(numeric_id)
        
        # Check if the file_id already contains site information (e.g., "Pitt_0050003")
        has_site_prefix = "_" in str(self.file_id) or "-" in str(self.file_id)
        
        # List of common site prefixes from ABIDE with potential index numbers
        site_prefixes = [
            "NYU", "SDSU", "STANFORD", "TRINITY", "UM_1", "UM_2", 
            "UCLA_1", "UCLA_2", "CALTECH", "LEUVEN_1", "LEUVEN_2", 
            "KKI", "PITT", "OHSU", "SBL", "YALE", "CMU", "MAXMUN"
        ]
        
        # Standard filename without site prefix (unlikely but check first)
        potential_filenames = [
            f"{self.file_id}_{self.derivative_name}.1D",  # Standard: file_id_derivative.1D
            f"{self.file_id}.1D",                         # Just subject ID
        ]
        
        # If ID doesn't already have site prefix, check with various site prefixes and formats
        if not has_site_prefix:
            # Try different padding lengths for the ID
            padding_lengths = [7, 5, 6, 8] if numeric_id.isdigit() else []
            padded_ids = [numeric_id.zfill(pad_len) for pad_len in padding_lengths] if numeric_id.isdigit() else []
            padded_ids.append(numeric_id)  # Also try without padding
            
            # Generate combinations of site prefixes and padded IDs
            for site in site_prefixes:
                for padded_id in padded_ids:
                    potential_filenames.extend([
                        f"{site}_{padded_id}_{self.derivative_name}.1D",
                        f"{site}/{site}_{padded_id}_{self.derivative_name}.1D"
                    ])
        
        # If file_id does have a site prefix, it might need just minor adjustments
        else:
            # Try adding the derivative name or different separators
            potential_filenames.extend([
                f"{self.file_id}_{self.derivative_name}.1D",
                # Extract parts and reconstruct with proper padding
                *[f"{part}_{numeric_id.zfill(7)}_{self.derivative_name}.1D" 
                  for part in self.file_id.split('_') if part and not part.isdigit()]
            ])
        
        logging.debug(f"Searching for file with {len(potential_filenames)} patterns including: {potential_filenames[:5]}...")
        
        # Also search in subdirectories if necessary
        search_dirs = [self.timeseries_dir]
        for root, dirs, _ in os.walk(self.timeseries_dir):
            for dir_name in dirs:
                search_dirs.append(os.path.join(root, dir_name))
        
        # Try each potential filename in each search directory
        for dir_path in search_dirs:
            for filename in potential_filenames:
                path = os.path.join(dir_path, filename)
                if os.path.exists(path):
                    self.timeseries_path = path
                    logging.info(f"Found timeseries file for {self.file_id} at {path}")
                    return
        
        # If no exact match was found, try using more flexible glob patterns
        for dir_path in search_dirs:
            # Try to find any file containing the numeric ID with the derivative name
            if str(numeric_id).isdigit():
                glob_patterns = [
                    os.path.join(dir_path, f"*{numeric_id}*{self.derivative_name}*.1D"),
                    os.path.join(dir_path, f"*_{numeric_id}*{self.derivative_name}*.1D"),
                    os.path.join(dir_path, f"*/*{numeric_id}*{self.derivative_name}*.1D")
                ]
                
                for pattern in glob_patterns:
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        self.timeseries_path = matching_files[0]
                        logging.info(f"Found timeseries file for {self.file_id} using glob pattern at {self.timeseries_path}")
                        return
        
        # If still no file was found, check for any .1D files that might match the site
        if has_site_prefix:
            site_part = self.file_id.split('_')[0]
            glob_pattern = os.path.join(self.timeseries_dir, f"{site_part}*{self.derivative_name}*.1D")
            matching_files = glob.glob(glob_pattern)
            if matching_files:
                # List found files for debugging
                logging.info(f"Found {len(matching_files)} files matching site {site_part}, example: {matching_files[0]}")
        
        # If no file was found, set a default path for error reporting
        self.timeseries_path = os.path.join(self.timeseries_dir, f"{self.file_id}_{self.derivative_name}.1D")
        logging.warning(f"Could not find timeseries file for {self.file_id}, tried {len(potential_filenames)} patterns, will use path: {self.timeseries_path}")

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
            # Try different ways to read the file based on extension
            file_ext = os.path.splitext(self.timeseries_path)[1].lower()
            
            if file_ext == '.csv':
                # Try comma delimiter for CSV files
                ts_data = pd.read_csv(self.timeseries_path, header=None).values
            else:
                # Use whitespace delimiter for .1D and .txt files
                ts_data = pd.read_csv(
                    self.timeseries_path, 
                    sep=r'\s+',     
                    header=None,    
                    comment='#'     
                ).values
            
            # Basic check if shape seems plausible (at least some TRs, expected regions)
            if ts_data.ndim != 2:
                logging.warning(f"Timeseries data from {self.timeseries_path} is not 2D, shape: {ts_data.shape}")
                return None
                
            if ts_data.shape[1] != self.expected_regions:
                # If transposed (more regions than TRs), try transposing
                if ts_data.shape[0] == self.expected_regions:
                    logging.warning(f"Transposing timeseries data from {self.timeseries_path}, shape was {ts_data.shape}")
                    ts_data = ts_data.T
                else:
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
def load_all_subject_data(config, phenotypic_file, results_dir, label_column='DX_GROUP'):
    """
    Loads, processes, and windows resting-state data for all included subjects from ABIDE.
    
    Args:
        config: Configuration dictionary
        phenotypic_file: Path to the main ABIDE phenotypic CSV file
        results_dir: Directory to save label encoder
        label_column: Column in phenotypic file to use as label (default: 'DX_GROUP' for classification,
                     can be clinical scores like 'ADOS_TOTAL' for regression)
        
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

    # --- Check label column exists ---
    if label_column not in pheno_df.columns:
        raise ValueError(f"Label column '{label_column}' not found in phenotypic data. Available columns: {pheno_df.columns.tolist()}")
    
    # --- Handle regression-specific preprocessing (for clinical scores) ---
    is_regression_task = label_column != 'DX_GROUP'
    
    if is_regression_task:
        # For regression, filter out subjects with missing values in the target column
        initial_count = len(pheno_df)
        pheno_df.dropna(subset=[label_column], inplace=True)
        missing_count = initial_count - len(pheno_df)
        if missing_count > 0:
            logging.info(f"Removed {missing_count} subjects with missing values in {label_column}")
        
        # Check for and potentially remove negative values (which often indicate missing data)
        exclude_negative = config.get('data', {}).get('exclude_negative_values', False)
        if exclude_negative:
            negative_mask = pheno_df[label_column] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                pheno_df = pheno_df[~negative_mask].copy()
                logging.info(f"Removed {negative_count} subjects with negative values in {label_column}")
        
        # Log statistics about the remaining values
        logging.info(f"Clinical score {label_column} statistics - Mean: {pheno_df[label_column].mean():.2f}, "
                    f"Std: {pheno_df[label_column].std():.2f}, Min: {pheno_df[label_column].min()}, "
                    f"Max: {pheno_df[label_column].max()}, Count: {len(pheno_df)}")
    
    # --- Process & Window Data ---
    # Create overall storage for windowed data
    all_windows = []
    all_labels = []
    all_subject_ids = []
    
    derivative_name = config['data'].get('derivative_name', 'rois_cc200')  # Helps with organizing results
    window_len = config['data']['seq_len']
    window_step = config['data'].get('window_step', 1)  # Step size for sliding window
    
    # Extract needed values from config
    timeseries_dir = config['paths']['regional_timeseries_dir']
    
    # Process each subject
    success_count = 0
    skipped_count = 0
    
    for idx, subject_row in pheno_df.iterrows():
        file_id = subject_row['FILE_ID']
        
        # Check if this subject has the label we're interested in
        if pd.isna(subject_row[label_column]):
            logging.warning(f"Subject {file_id} missing {label_column} value, skipping.")
            skipped_count += 1
            continue
        
        # Get the appropriate label
        if is_regression_task:
            # For regression, use the raw value (already verified not NaN above)
            subject_label = float(subject_row[label_column])
        else:
            # For classification, use the DX_GROUP (1=ASD, 2=TDC) and map to 0,1 for easier use
            subject_label = subject_row['DX_GROUP']
            # Convert from 1/2 to 0/1 for ASD/TDC
            subject_label = 0 if subject_label == 1 else 1
        
        # Process timeseries
        processor = AbideTimeseriesProcessor(config, file_id)
        timeseries = processor.process()
        
        if timeseries is None:
            logging.warning(f"Failed to process timeseries for subject {file_id}, skipping.")
            skipped_count += 1
            continue
            
        # Create windows from timeseries, using subject ID for grouping
        windows, window_subject_ids = create_resting_state_windows(
            timeseries, file_id, seq_len=window_len, step=window_step
        )
        
        if windows is None:
            logging.warning(f"No valid windows created for subject {file_id}, skipping.")
            skipped_count += 1
            continue
        
        # Create labels for all windows from this subject (same label for each window)
        window_labels = np.full(len(windows), subject_label, dtype=np.float32 if is_regression_task else np.int64)
        
        # Add to master lists
        all_windows.append(windows)
        all_labels.append(window_labels)
        all_subject_ids.append(window_subject_ids)
        
        success_count += 1
        
        if idx % 10 == 0:
            logging.info(f"Processed {idx+1}/{len(pheno_df)} subjects...")
    
    if success_count == 0:
        raise ValueError("No subjects successfully processed! Check your data files and filters.")
        
    logging.info(f"Successfully processed {success_count} subjects, skipped {skipped_count}")
    
    # Combine all arrays
    X_all = np.vstack(all_windows)
    y_all = np.concatenate(all_labels)
    groups_all = np.concatenate(all_subject_ids)
    
    # Only create label encoder for classification task
    label_encoder = None
    if not is_regression_task:
        # For classification, we use a label encoder
        label_encoder = LabelEncoder().fit(y_all)
        encoder_path = os.path.join(results_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Transform labels
        y_all = label_encoder.transform(y_all)
    
    logging.info(f"Final dataset shape - X: {X_all.shape}, y: {y_all.shape}, groups: {groups_all.shape}")
    logging.info(f"Number of unique subjects: {len(np.unique(groups_all))}")
    
    if is_regression_task:
        logging.info(f"Target variable ({label_column}) stats - Mean: {np.mean(y_all):.2f}, "
                    f"Std: {np.std(y_all):.2f}, Min: {np.min(y_all)}, Max: {np.max(y_all)}")
    else:
        # Count number of samples in each class
        class_counts = np.bincount(y_all)
        for i, count in enumerate(class_counts):
            class_name = label_encoder.inverse_transform([i])[0]
            logging.info(f"Class {i} ({class_name}): {count} samples")
    
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


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for timeseries data with support for both classification and regression tasks"""
    def __init__(self, data, labels=None, regression_targets=None):
        """
        Args:
            data: List or array of timeseries data tensors
            labels: List or array of class labels (for classification tasks)
            regression_targets: List or array of target values (for regression tasks)
        """
        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = [torch.tensor(d, dtype=torch.float32) if not isinstance(d, torch.Tensor) else d for d in data]
        
        self.is_regression = regression_targets is not None
        
        if self.is_regression:
            if isinstance(regression_targets, np.ndarray):
                self.targets = torch.tensor(regression_targets, dtype=torch.float32)
            else:
                self.targets = [torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t 
                              for t in regression_targets]
        else:
            if isinstance(labels, np.ndarray):
                self.targets = torch.tensor(labels, dtype=torch.long)
            else:
                self.targets = [torch.tensor(l, dtype=torch.long) if not isinstance(l, torch.Tensor) else l 
                              for l in labels]
        
        logging.debug(f"TimeSeriesDataset created with {len(self.data)} samples for {'regression' if self.is_regression else 'classification'} task")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def load_timeseries_data(config, subject_ids=None, task_type='classification'):
    """
    Load time series data for specified subjects.
    
    Args:
        config: Configuration dictionary
        subject_ids: List of subject IDs to include (None to load all)
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Tuple containing data dictionary and clinical data DataFrame
    """
    # Extract paths and parameters from config
    base_dir = config.get('paths', {}).get('base_dir', '')
    timeseries_dir = config.get('paths', {}).get('regional_timeseries_dir', '')
    phenotypic_file = config.get('paths', {}).get('phenotypic_file', '')
    clinical_data_path = os.path.join(base_dir, phenotypic_file) if base_dir and phenotypic_file else ''
    
    # Check if clinical data path exists
    if not clinical_data_path or not os.path.exists(clinical_data_path):
        clinical_data_path = config.get('experiment', {}).get('data_path', '')
        if os.path.isdir(clinical_data_path):
            # Look for phenotypic file in the data_path directory
            potential_path = os.path.join(clinical_data_path, phenotypic_file)
            if os.path.exists(potential_path):
                clinical_data_path = potential_path
                logging.info(f"Using phenotypic file: {clinical_data_path}")
            else:
                # Look for any CSV files in the directory
                csv_files = glob.glob(os.path.join(clinical_data_path, "*.csv"))
                if csv_files:
                    clinical_data_path = csv_files[0]
                    logging.info(f"Found CSV file: {clinical_data_path}")
                else:
                    raise FileNotFoundError(f"No CSV files found in {clinical_data_path}")
        else:
            logging.warning(f"Using data path from experiment config: {clinical_data_path}")
    
    # Load clinical data
    try:
        clinical_data = pd.read_csv(clinical_data_path)
        logging.info(f"Loaded clinical data from {clinical_data_path}, shape: {clinical_data.shape}")
    except Exception as e:
        logging.error(f"Error loading clinical data from {clinical_data_path}: {e}")
        raise
    
    # Filter by included sites if specified in config
    included_sites = config.get('qc', {}).get('included_sites', None)
    if included_sites and 'SITE_ID' in clinical_data.columns:
        original_count = len(clinical_data)
        clinical_data = clinical_data[clinical_data['SITE_ID'].isin(included_sites)]
        filtered_count = len(clinical_data)
        logging.info(f"Filtered by included sites: {included_sites}. Kept {filtered_count}/{original_count} subjects.")
    
    # Determine subject ID column
    subject_id_col = config.get('data', {}).get('subject_id_col', 'subject')
    if subject_id_col not in clinical_data.columns:
        # Try to find a subject ID column
        potential_cols = ['subject', 'Subject', 'SUBJECT', 'SUB_ID', 'Subject_ID', 'subject_id', 'FILE_ID']
        for col in potential_cols:
            if col in clinical_data.columns:
                subject_id_col = col
                logging.warning(f"Using '{subject_id_col}' as subject ID column")
                break
        else:
            raise ValueError(f"Subject ID column not found in clinical data. Available columns: {clinical_data.columns.tolist()}")
    
    # Look for a file ID column (if different from subject ID)
    file_id_col = config.get('data', {}).get('file_id_col', None)
    if not file_id_col or file_id_col not in clinical_data.columns:
        # Try to find a file ID column if it's different from subject_id_col
        potential_file_cols = ['FILE_ID', 'file_id', 'File_ID']
        for col in potential_file_cols:
            if col in clinical_data.columns and col != subject_id_col:
                file_id_col = col
                logging.info(f"Using '{file_id_col}' for timeseries file matching")
                break
        else:
            # If no separate file ID column found, use subject ID
            file_id_col = subject_id_col
            logging.info(f"Using subject ID column '{subject_id_col}' for file matching")
    
    # Handle the case where subject_ids are provided
    filtered_clinical_data = clinical_data
    if subject_ids is not None and len(subject_ids) > 0:
        # Convert subject_ids to the right type for comparison
        if subject_id_col in clinical_data.columns:
            # Check if we need to convert the subject IDs
            sample_col_value = clinical_data[subject_id_col].iloc[0]
            if isinstance(sample_col_value, (int, float)) and all(isinstance(sid, str) for sid in subject_ids):
                # Convert string subject IDs to numeric
                try:
                    numeric_subject_ids = [float(sid) for sid in subject_ids]
                    logging.info(f"Converting string subject IDs to numeric for matching: {subject_ids} -> {numeric_subject_ids}")
                    subject_ids = numeric_subject_ids
                except ValueError:
                    logging.warning(f"Could not convert all subject IDs to numeric, using as is: {subject_ids}")
            
            # Try direct matching first
            filtered_clinical_data = clinical_data[clinical_data[subject_id_col].isin(subject_ids)]
            logging.info(f"Filtered clinical data to {len(filtered_clinical_data)} subjects using direct matching")
            
            # If no matches and we have FILE_ID column, try pattern matching
            if len(filtered_clinical_data) == 0 and file_id_col in clinical_data.columns and file_id_col != subject_id_col:
                logging.info(f"No direct matches found, trying pattern matching on '{file_id_col}'")
                
                # Create a mask for pattern matching
                mask = pd.Series(False, index=clinical_data.index)
                for sid in subject_ids:
                    # Try different pattern formats
                    for pattern in [f"_{sid}", f"_{sid}_", str(sid)]:
                        pattern_mask = clinical_data[file_id_col].astype(str).str.contains(pattern, regex=False)
                        mask = mask | pattern_mask
                
                # Apply the mask
                filtered_clinical_data = clinical_data[mask]
                logging.info(f"Filtered clinical data to {len(filtered_clinical_data)} subjects using pattern matching")
                
                if len(filtered_clinical_data) == 0:
                    logging.warning(f"Could not find any subjects matching IDs: {subject_ids}")
        else:
            logging.warning(f"Subject ID column '{subject_id_col}' not found in clinical data, using all subjects")
    
    # Update clinical data to the filtered version
    clinical_data = filtered_clinical_data
    logging.info(f"Using {len(clinical_data)} subjects for data loading")
    
    # For regression tasks, ensure the regression target column exists
    if task_type == 'regression':
        regression_target = config.get('data', {}).get('regression_target')
        if not regression_target:
            logging.error(f"Regression target not found in config: {config.get('data', {})}")
            regression_target = config.get('data', {}).get('target_column')
            if regression_target:
                logging.warning(f"Using 'target_column' as regression target: {regression_target}")
            
        if not regression_target or regression_target not in clinical_data.columns:
            raise ValueError(f"Regression target '{regression_target}' not found in clinical data. Available columns: {clinical_data.columns.tolist()}")
        
        # Remove subjects with missing or invalid target values
        valid_mask = ~clinical_data[regression_target].isna()
        if valid_mask.sum() < len(clinical_data):
            logging.warning(f"Removing {len(clinical_data) - valid_mask.sum()} subjects with missing {regression_target} values")
            clinical_data = clinical_data[valid_mask]
        
        # Optional: Remove subjects with negative values (often used as missing indicators)
        exclude_negative = config.get('data', {}).get('exclude_negative_values', False)
        if exclude_negative:
            neg_mask = clinical_data[regression_target] < 0
            if neg_mask.sum() > 0:
                logging.warning(f"Removing {neg_mask.sum()} subjects with negative {regression_target} values")
                clinical_data = clinical_data[~neg_mask]
        
        # Log statistics about the target variable
        if len(clinical_data) > 0:
            logging.info(f"Target variable ({regression_target}) stats: min={clinical_data[regression_target].min()}, "
                        f"max={clinical_data[regression_target].max()}, mean={clinical_data[regression_target].mean():.2f}, "
                        f"std={clinical_data[regression_target].std():.2f}")
        else:
            logging.warning(f"No subjects with valid {regression_target} values after filtering")
    
    # Load timeseries data
    seq_length = config.get('data', {}).get('sequence_length', 100)
    # Try alternative key if sequence_length not found
    if not seq_length:
        seq_length = config.get('data', {}).get('seq_len', 30)
        if seq_length:
            logging.warning(f"Using 'seq_len' instead of 'sequence_length': {seq_length}")
    
    window_step = config.get('data', {}).get('window_step', 10)
    
    logging.info(f"Using sequence length: {seq_length}, window step: {window_step}")
    
    # Initialize dictionary to store subject data
    subject_data_dict = {}
    
    # Get unique subject IDs
    subjects = clinical_data[subject_id_col].unique()
    logging.info(f"Processing {len(subjects)} subjects")
    
    # Process each subject
    success_count = 0
    skip_count = 0
    
    for subject_id in subjects:
        try:
            # Get file_id from the clinical data if it's different from subject_id
            if file_id_col != subject_id_col:
                file_id_data = clinical_data[clinical_data[subject_id_col] == subject_id][file_id_col]
                if len(file_id_data) > 0 and not pd.isna(file_id_data.iloc[0]) and file_id_data.iloc[0] != 'no_filename':
                    file_id = file_id_data.iloc[0]
                    logging.debug(f"Using FILE_ID '{file_id}' for subject {subject_id}")
                else:
                    file_id = subject_id
                    logging.debug(f"No valid FILE_ID found for subject {subject_id}, using subject ID")
            else:
                file_id = subject_id
            
            # Convert file_id to string if it's numeric
            if isinstance(file_id, (int, float)):
                file_id = str(int(file_id))
            
            # Construct processor
            processor = AbideTimeseriesProcessor(config, file_id)
            timeseries = processor.process()
            
            if timeseries is None:
                logging.warning(f"Failed to process timeseries for subject {subject_id}, skipping")
                skip_count += 1
                continue
            
            # Create windows
            windows, _ = create_resting_state_windows(
                timeseries, subject_id, seq_len=seq_length, step=window_step
            )
            
            if windows is None or len(windows) == 0:
                logging.warning(f"No valid windows created for subject {subject_id}, skipping")
                skip_count += 1
                continue
            
            # Store windows for this subject
            subject_data_dict[subject_id] = windows
            success_count += 1
            
        except Exception as e:
            logging.error(f"Error processing subject {subject_id}: {e}")
            skip_count += 1
            continue
    
    logging.info(f"Successfully loaded data for {success_count} subjects, skipped {skip_count} subjects")
    if success_count == 0:
        raise ValueError("No subjects were successfully processed. Check your timeseries paths and configuration.")
        
    return subject_data_dict, clinical_data


def create_data_loaders(subject_data_dict, clinical_data, config, fold_indices=None, task_type='classification'):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        subject_data_dict: Dictionary mapping subject IDs to their windowed timeseries data
        clinical_data: DataFrame containing clinical data
        config: Configuration dictionary
        fold_indices: Dictionary containing train/val/test indices or subject IDs
        task_type: Type of task ('classification' or 'regression')
        
    Returns:
        Tuple containing train, validation, and test data loaders
    """
    # Extract parameters from config
    batch_size = config.get('training', {}).get('batch_size', 32)
    validation_batch_size = config.get('training', {}).get('validation_batch_size', batch_size)
    shuffle = config.get('training', {}).get('shuffle', True)
    num_workers = config.get('training', {}).get('num_workers', 4)
    subject_id_col = config.get('data', {}).get('subject_id_col', 'subject')
    
    # Determine target column based on task type
    if task_type == 'regression':
        target_column = config.get('data', {}).get('regression_target')
        if target_column not in clinical_data.columns:
            raise ValueError(f"Regression target '{target_column}' not found in clinical data")
    else:
        target_column = config.get('data', {}).get('label_column', 'DX_GROUP')
        if target_column not in clinical_data.columns:
            raise ValueError(f"Label column '{target_column}' not found in clinical data")
    
    # Extract subjects for each set based on fold indices
    if fold_indices is not None:
        # Check if we have direct subject lists (new approach)
        if 'train_subjects' in fold_indices:
            train_subjects = fold_indices['train_subjects']
            val_subjects = fold_indices['val_subjects'] 
            test_subjects = fold_indices['test_subjects']
            
        # Legacy approach with DataFrame indices
        elif all(key in fold_indices for key in ['train', 'val', 'test']):
            try:
                train_indices = fold_indices['train']
                val_indices = fold_indices['val']
                test_indices = fold_indices['test']
                
                train_subjects = clinical_data.iloc[train_indices][subject_id_col].unique()
                val_subjects = clinical_data.iloc[val_indices][subject_id_col].unique()
                test_subjects = clinical_data.iloc[test_indices][subject_id_col].unique()
            except IndexError:
                # Fallback if indices are out of bounds
                logging.warning("Index error with fold_indices, using default split")
                subjects = clinical_data[subject_id_col].unique()
                np.random.shuffle(subjects)
                n_subjects = len(subjects)
                n_train = int(0.7 * n_subjects)
                n_val = int(0.15 * n_subjects)
                
                train_subjects = subjects[:n_train]
                val_subjects = subjects[n_train:n_train+n_val]
                test_subjects = subjects[n_train+n_val:]
        else:
            # Invalid fold_indices format
            raise ValueError("fold_indices must contain either 'train_subjects' or 'train'/'val'/'test' keys")
    else:
        # Default split if fold indices not provided (70/15/15)
        subjects = clinical_data[subject_id_col].unique()
        np.random.shuffle(subjects)
        n_subjects = len(subjects)
        n_train = int(0.7 * n_subjects)
        n_val = int(0.15 * n_subjects)
        
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train+n_val]
        test_subjects = subjects[n_train+n_val:]
    
    # Collect data and targets for each set
    train_data = []
    val_data = []
    test_data = []
    
    train_targets = []
    val_targets = []
    test_targets = []
    
    # Collect data and targets for training set
    for subject_id in train_subjects:
        if subject_id in subject_data_dict:
            windows = subject_data_dict[subject_id]
            for window in windows:
                train_data.append(window)
                
                # Get target value for this subject
                subject_rows = clinical_data[clinical_data[subject_id_col] == subject_id]
                if len(subject_rows) > 0:
                    target = subject_rows[target_column].values[0]
                    if task_type == 'regression':
                        target = float(target)
                    else:
                        target = int(target)
                    
                    train_targets.append(target)
                else:
                    logging.warning(f"No clinical data found for subject {subject_id}, skipping window")
                    continue
    
    # Collect data and targets for validation set
    for subject_id in val_subjects:
        if subject_id in subject_data_dict:
            windows = subject_data_dict[subject_id]
            for window in windows:
                val_data.append(window)
                
                # Get target value for this subject
                subject_rows = clinical_data[clinical_data[subject_id_col] == subject_id]
                if len(subject_rows) > 0:
                    target = subject_rows[target_column].values[0]
                    if task_type == 'regression':
                        target = float(target)
                    else:
                        target = int(target)
                    
                    val_targets.append(target)
                else:
                    logging.warning(f"No clinical data found for subject {subject_id}, skipping window")
                    continue
    
    # Collect data and targets for test set
    for subject_id in test_subjects:
        if subject_id in subject_data_dict:
            windows = subject_data_dict[subject_id]
            for window in windows:
                test_data.append(window)
                
                # Get target value for this subject
                subject_rows = clinical_data[clinical_data[subject_id_col] == subject_id]
                if len(subject_rows) > 0:
                    target = subject_rows[target_column].values[0]
                    if task_type == 'regression':
                        target = float(target)
                    else:
                        target = int(target)
                    
                    test_targets.append(target)
                else:
                    logging.warning(f"No clinical data found for subject {subject_id}, skipping window")
                    continue
    
    # Create PyTorch datasets
    if task_type == 'regression':
        train_dataset = TimeSeriesDataset(train_data, regression_targets=train_targets)
        val_dataset = TimeSeriesDataset(val_data, regression_targets=val_targets)
        test_dataset = TimeSeriesDataset(test_data, regression_targets=test_targets)
    else:
        train_dataset = TimeSeriesDataset(train_data, labels=train_targets)
        val_dataset = TimeSeriesDataset(val_data, labels=val_targets)
        test_dataset = TimeSeriesDataset(test_data, labels=test_targets)
    
    # Create and return data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Log data sizes
    logging.info(f"Created data loaders for {task_type} task:")
    logging.info(f"  - Training: {len(train_dataset)} samples from {len(train_subjects)} subjects")
    logging.info(f"  - Validation: {len(val_dataset)} samples from {len(val_subjects)} subjects")
    logging.info(f"  - Test: {len(test_dataset)} samples from {len(test_subjects)} subjects")
    
    if task_type == 'regression':
        train_targets_np = np.array(train_targets)
        logging.info(f"  - Target stats (train): min={train_targets_np.min():.2f}, max={train_targets_np.max():.2f}, "
                    f"mean={train_targets_np.mean():.2f}, std={train_targets_np.std():.2f}")
    else:
        class_counts = np.bincount(np.array(train_targets))
        for i, count in enumerate(class_counts):
            logging.info(f"  - Class {i}: {count} samples in training set")
    
    return train_loader, val_loader, test_loader