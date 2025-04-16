import numpy as np
import logging
from typing import Tuple, List

# Setup logger
logger = logging.getLogger(__name__)

def compute_static_fc(timeseries_window: np.ndarray, subject_id: str = 'Unknown') -> np.ndarray:
    """
    Computes the static functional connectivity matrix for a single time series window
    and returns its flattened upper triangle (excluding the diagonal).

    Args:
        timeseries_window (np.ndarray): Input window of shape (seq_len, num_regions).
        subject_id (str): Identifier for the subject this window belongs to (for logging).

    Returns:
        np.ndarray: Flattened upper triangle of the correlation matrix (1D vector).
                   Returns None if calculation fails.
    """
    if timeseries_window is None or timeseries_window.shape[0] < 2 or timeseries_window.shape[1] < 2:
        logging.warning(f"[{subject_id}] Skipping FC calculation for invalid window shape: {timeseries_window.shape if timeseries_window is not None else 'None'}")
        # Return None to indicate failure (changed from empty array)
        return None

    try:
        # Calculate Pearson correlation coefficient matrix
        # Ensure data is float and handle potential NaNs/Infs from previous steps
        timeseries_window = np.nan_to_num(timeseries_window.astype(np.float64))

        # Check for zero variance columns - correlation is undefined
        std_devs = np.std(timeseries_window, axis=0)
        if np.any(std_devs == 0):
            # Find indices of zero variance columns for detailed logging
            zero_var_indices = np.where(std_devs == 0)[0]
            logging.warning(f"[{subject_id}] Window contains regions with zero variance (indices: {zero_var_indices}). Cannot compute correlations. Skipping window.")
            # Return None to indicate failure
            return None

        correlation_matrix = np.corrcoef(timeseries_window, rowvar=False)

        # Handle potential NaNs if correlation is undefined (e.g., constant signals after nan_to_num)
        # This shouldn't happen if zero variance is caught, but as a safeguard:
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0) # Replace NaN with 0

        # Extract the upper triangle indices (excluding the diagonal k=1)
        num_regions = correlation_matrix.shape[0]
        upper_triangle_indices = np.triu_indices(num_regions, k=1)

        # Flatten the upper triangle
        flat_fc = correlation_matrix[upper_triangle_indices]

        # Ensure float32 for PyTorch
        return flat_fc.astype(np.float32)

    except Exception as e:
        logging.error(f"[{subject_id}] Error computing FC for window shape {timeseries_window.shape}: {e}", exc_info=True)
        # Return None on other errors as well
        return None


def compute_fc_matrices(timeseries_data: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute full connectivity matrices from timeseries data.
    
    Args:
        timeseries_data (np.ndarray): Array of timeseries data with shape 
                                      (n_subjects, n_timepoints, n_regions)
    
    Returns:
        Tuple[np.ndarray, List[int]]: 
            - Array of connectivity matrices with shape (n_subjects, n_regions, n_regions).
            - List of integer indices for subjects found to have zero variance in at least one region.
    """
    if len(timeseries_data.shape) != 3:
        raise ValueError(f"Expected 3D timeseries data with shape (n_subjects, n_timepoints, n_regions), "
                         f"got shape {timeseries_data.shape}")
    
    n_subjects, n_timepoints, n_regions = timeseries_data.shape
    fc_matrices = []
    subjects_with_zero_variance = []
    
    logging.info(f"Computing connectivity matrices for {n_subjects} subjects...")
    
    for i in range(n_subjects):
        # Get timeseries for this subject
        subject_ts = timeseries_data[i]
        has_zero_variance = False
        
        try:
            # Compute correlation matrix
            subject_ts = np.nan_to_num(subject_ts.astype(np.float64))
            
            # Check for zero variance regions
            std_devs = np.std(subject_ts, axis=0)
            if np.any(std_devs == 0):
                has_zero_variance = True
                subjects_with_zero_variance.append(i)
                zero_var_indices = np.where(std_devs == 0)[0]
                logging.warning(f"Subject {i} has regions with zero variance (indices: {zero_var_indices}). "
                               f"Marking for exclusion, but computing matrix with zeros for now.")
                
                # Compute correlation matrix with zeros for affected regions
                corr_matrix = np.zeros((n_regions, n_regions), dtype=np.float32)
                
                # Get indices of non-zero variance regions
                valid_indices = np.where(std_devs > 0)[0]
                if len(valid_indices) > 1:
                    # Compute correlations for valid regions
                    valid_ts = subject_ts[:, valid_indices]
                    valid_corr = np.corrcoef(valid_ts, rowvar=False)
                    
                    # Place valid correlations in the full matrix
                    for i1, idx1 in enumerate(valid_indices):
                        for i2, idx2 in enumerate(valid_indices):
                            corr_matrix[idx1, idx2] = valid_corr[i1, i2]
            else:
                # All regions have variance, compute full correlation matrix
                corr_matrix = np.corrcoef(subject_ts, rowvar=False)
            
            # Handle any NaNs and ensure dtype
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            fc_matrices.append(corr_matrix.astype(np.float32))
            
        except Exception as e:
            logging.error(f"Error computing connectivity matrix for subject {i}: {e}")
            # Add a matrix of zeros as a placeholder
            fc_matrices.append(np.zeros((n_regions, n_regions), dtype=np.float32))
            if i not in subjects_with_zero_variance: 
                 subjects_with_zero_variance.append(i)
    
    return np.array(fc_matrices), subjects_with_zero_variance


def preprocess_windows_to_fc(X_windows: np.ndarray, y_windows: np.ndarray, groups_windows: np.ndarray):
    """
    Converts a batch of time series windows into flattened static FC features.

    Args:
        X_windows (np.ndarray): Array of time series windows, shape (num_windows, seq_len, num_regions).
        y_windows (np.ndarray): Array of labels for each window.
        groups_windows (np.ndarray): Array of group/subject IDs for each window.

    Returns:
        tuple: (fc_features, y_fc, groups_fc) where fc_features is an array of
               shape (num_valid_windows, num_fc_features), and y/groups are filtered
               to match the valid windows.
    """
    if X_windows.ndim != 3 or X_windows.shape[0] == 0:
        logging.error(f"Invalid input shape for X_windows: {X_windows.shape}. Expected (num_windows, seq_len, num_regions).")
        return np.array([]), np.array([]), np.array([])
        
    num_windows, seq_len, num_regions = X_windows.shape
    logging.info(f"Starting FC preprocessing for {num_windows} windows ({seq_len} timesteps, {num_regions} regions)...")
    fc_features_list = []
    valid_indices = []

    expected_fc_size = (num_regions * (num_regions - 1)) // 2
    if expected_fc_size <= 0:
        logging.error(f"Cannot compute FC features for {num_regions} regions.")
        return np.array([]), np.array([]), np.array([])

    for i in range(num_windows):
        subject_id = str(groups_windows[i]) if i < len(groups_windows) else 'Unknown'
        fc_vec = compute_static_fc(X_windows[i], subject_id=subject_id)
        # Only keep windows where FC calculation succeeded (returned a vector, not None)
        if fc_vec is not None:
            # Double check size just in case, though None should catch most issues
            if fc_vec.size == expected_fc_size:
                fc_features_list.append(fc_vec)
                valid_indices.append(i)
            else:
                logging.warning(f"Window {i} FC calculation resulted in unexpected size: {fc_vec.size}, expected: {expected_fc_size}. Skipping.")
        # No need for an else here, compute_static_fc logs the specific reason for returning None

    if not fc_features_list:
        logging.error("No valid FC features could be computed from the input windows.")
        return np.array([]), np.array([]), np.array([])

    fc_features = np.stack(fc_features_list, axis=0)
    y_fc = y_windows[valid_indices]
    groups_fc = groups_windows[valid_indices]

    logging.info(f"FC preprocessing complete. Generated {fc_features.shape[0]} valid FC feature vectors.")
    logging.info(f"Shape of FC features: {fc_features.shape}")

    return fc_features, y_fc, groups_fc

"""
Example Usage:

# Assume X_all, y_all, groups_all are loaded windowed data
# X_all shape: (num_total_windows, seq_len, num_regions)

# fc_features, y_fc, groups_fc = preprocess_windows_to_fc(X_all, y_all, groups_all)

# Now fc_features can be used as input to an MLP or other models.
# fc_features shape: (num_valid_windows, num_fc_features)
"""