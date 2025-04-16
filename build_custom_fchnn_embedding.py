import os
import pandas as pd
import numpy as np
import yaml
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from src.utils import set_seed, get_device

import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Import connattractor ---
try:
    from connattractor import network, analysis, utils
except ImportError:
    print("ERROR: connattractor library not found. Install: pip install connattractor")
    exit()

def load_subject_filepaths(config):
    """Loads phenotype data just to find valid subject TS file paths."""
    data_cfg = config['data_params']
    phenotype_file = data_cfg['phenotype_file']
    sub_id_col = data_cfg['sub_id_col']
    site_id_col = data_cfg.get('site_id_col', 'SITE_ID')
    target_col = data_cfg['target_col'] # Needed if filtering by controls
    region_dir = data_cfg['region_dir']
    connectome_subject_group = config['embedding_build_params'].get('connectome_subjects', 'all')

    print(f"Loading phenotype data from {phenotype_file} to identify subjects...")
    try:
        # Load only necessary columns
        usecols = [sub_id_col, site_id_col, target_col]
        df_pheno_full = pd.read_csv(phenotype_file, usecols=usecols)

        # Map target: 0=Control, 1=ASD (as done in preprocessing)
        df_pheno_full[target_col] = df_pheno_full[target_col].replace([-9999, -9999.0, ''], np.nan)
        df_pheno_full[target_col] = pd.to_numeric(df_pheno_full[target_col], errors='coerce')
        df_pheno_full[target_col] = df_pheno_full[target_col].apply(lambda x: 0 if x == 2 else 1 if x == 1 else -1)
        df_pheno_full = df_pheno_full[df_pheno_full[target_col] != -1] # Keep only valid targets

    except Exception as e:
        print(f"Error reading phenotype file: {e}"); return None

    subject_files = {}
    print(f"Checking TS files in {region_dir} for group '{connectome_subject_group}'...")
    for index, row in tqdm(df_pheno_full.iterrows(), total=len(df_pheno_full), desc="Finding Files"):
        sub_id = None
        site_id = None
        try:
            sub_id = int(row[sub_id_col])
            site_id = row[site_id_col]
            target = row[target_col]

            # Filter by group if needed
            if connectome_subject_group == 'controls_only' and target != 0:
                continue
            # Add elif for 'asd_only' if needed

            filename = f"{site_id}_{sub_id:07d}_rois_cc200.1D"
            fpath = os.path.join(region_dir, filename)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                subject_files[sub_id] = fpath # Store path

        except (ValueError, TypeError, KeyError): continue # Skip invalid IDs/rows
        except Exception as e_inner: print(f"Error checking row {index}: {e_inner}"); continue

    print(f"Found {len(subject_files)} valid TS files for selected group.")
    if not subject_files: print("Error: No subject files found for building connectome."); return None
    return subject_files

def calculate_static_fc(timeseries_data):
    """Calculates static Pearson correlation matrix."""
    if not isinstance(timeseries_data, np.ndarray): timeseries_data = np.array(timeseries_data)
    if not np.issubdtype(timeseries_data.dtype, np.number): return None
    if not np.all(np.isfinite(timeseries_data)): timeseries_data = np.nan_to_num(timeseries_data)
    if timeseries_data.shape[0] < 2: return None # Need >1 time point

    try:
         # Use errstate to suppress warnings about zero variance locally
         with np.errstate(divide='ignore', invalid='ignore'):
              correlation_matrix = np.corrcoef(timeseries_data.T)
         # Handle NaNs *after* calculation
         correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
         # Ensure diagonal is 1 (sometimes becomes NaN then 0)
         np.fill_diagonal(correlation_matrix, 1.0)
    except Exception as e: print(f"Error in corrcoef: {e}"); return None
    if not np.all(np.isfinite(correlation_matrix)): return None
    return correlation_matrix

def main():
    build_start_time = time.time()
    # --- Load Config ---
    config_filename = 'config_custom_fchnn.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Error loading {config_filename}: {e}"); return

    data_cfg = config['data_params']
    build_cfg = config['embedding_build_params']
    run_cfg = config['run_params'] # For seed
    num_regions = data_cfg['num_regions']
    embedding_dir = data_cfg['embedding_output_dir']
    embedding_file = os.path.join(embedding_dir, 'custom_chnn_embedding.pkl')

    # --- Set Seed ---
    set_seed(run_cfg['seed']) # Use connattractor's seed setter if needed

    # --- Find Subject Files for Connectome ---
    subject_files_dict = load_subject_filepaths(config)
    if subject_files_dict is None: return

    # --- Calculate Average Connectome ---
    print("Calculating individual connectomes and averaging...")
    all_fc_matrices = []
    ts_scaler = StandardScaler()
    failed_fc_count = 0

    for sub_id, fpath in tqdm(subject_files_dict.items(), desc="Calculating FC"):
        try:
            ts_raw = np.loadtxt(fpath)
            if ts_raw.ndim != 2 or ts_raw.shape[1] != num_regions or ts_raw.shape[0] < 5:
                failed_fc_count += 1; continue
            ts_normalized = ts_scaler.fit_transform(ts_raw)
            fc_matrix = calculate_static_fc(ts_normalized)
            if fc_matrix is not None and fc_matrix.shape == (num_regions, num_regions):
                # Apply Fisher Z-transform before averaging
                all_fc_matrices.append(np.arctanh(np.clip(fc_matrix, -0.9999, 0.9999)))
            else:
                failed_fc_count += 1
        except Exception as e:
            print(f"Error processing FC for {sub_id}: {e}")
            failed_fc_count += 1

    if not all_fc_matrices:
        print("Error: Could not calculate any valid FC matrices."); return
    print(f"Successfully calculated FC for {len(all_fc_matrices)} subjects (skipped {failed_fc_count}).")

    # Average the Z-transformed matrices
    mean_z_fc = np.mean(all_fc_matrices, axis=0)
    # Inverse Fisher Z-transform
    group_fc_mtx = np.tanh(mean_z_fc)
    # Ensure diagonal is 1 and matrix is symmetric
    np.fill_diagonal(group_fc_mtx, 1.0)
    group_fc_mtx = (group_fc_mtx + group_fc_mtx.T) / 2
    print(f"Group FC matrix calculated. Shape: {group_fc_mtx.shape}")

    # --- Construct Hopfield Network ---
    print("Constructing Hopfield network...")
    hopnet = network.Hopfield(group_fc_mtx)

    # --- Simulate Activations ---
    sim_params = {k.replace('simulation_', ''): v for k, v in build_cfg.items() if k.startswith('simulation_')}
    sim_params['random_state'] = run_cfg['seed'] # Ensure reproducibility
    print(f"Running stochastic relaxation with params: {sim_params}")
    # Note: analysis.simulate_activations expects the connectivity matrix directly
    chnn_state_space = analysis.simulate_activations(group_fc_mtx, **sim_params)
    print("Simulation finished.")

    # --- Create Embedding ---
    embed_params = {k.replace('embedding_', ''): v for k, v in build_cfg.items() if k.startswith('embedding_')}
    embed_params['random_state'] = run_cfg['seed']
    print(f"Creating embedding with params: {embed_params}")
    chnn_projection = analysis.create_embeddings(chnn_state_space, **embed_params)
    print("Embedding created.")
    print(f" Found {chnn_projection.attractor_coords.shape[0]} attractors.")

    # --- Save Embedding Object ---
    os.makedirs(embedding_dir, exist_ok=True)
    print(f"Saving embedding object to: {embedding_file}")
    try:
        with open(embedding_file, 'wb') as f:
            pickle.dump(chnn_projection, f)
        print("Embedding saved successfully.")
    except Exception as e:
        print(f"Error saving embedding object: {e}")

    build_end_time = time.time()
    print(f"--- Embedding Build Finished ({build_end_time - build_start_time:.2f} seconds) ---")

if __name__ == "__main__":
    main()