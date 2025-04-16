import os
import pandas as pd
import numpy as np
import yaml
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Import connattractor and helpers ---
try:
    from connattractor import utils as fchnn_utils # For loading if needed
    # Import helper functions from the other baseline script if they exist
    # Or copy them here:
    def assign_states_to_timepoints(projected_coords, attractor_coords):
        num_timepoints = projected_coords.shape[0]
        num_states = attractor_coords.shape[0]
        state_assignments = np.zeros(num_timepoints, dtype=int)
        for t in range(num_timepoints):
            point = projected_coords[t, :2]
            distances = np.linalg.norm(attractor_coords - point, axis=1)
            state_assignments[t] = np.argmin(distances)
        return state_assignments

    def calculate_fractional_occupancy(state_assignments, num_states):
        counts = np.bincount(state_assignments, minlength=num_states)
        total_time = len(state_assignments)
        return counts / total_time if total_time > 0 else np.zeros(num_states)

    def calculate_dwell_times(state_assignments, num_states):
        if len(state_assignments) == 0: return np.zeros(num_states)
        avg_dwell_times = np.zeros(num_states); state_counts = np.zeros(num_states, dtype=int)
        current_state = state_assignments[0]; current_dwell = 0
        for i in range(len(state_assignments)):
            if state_assignments[i] == current_state: current_dwell += 1
            else:
                avg_dwell_times[current_state] += current_dwell; state_counts[current_state] += 1
                current_state = state_assignments[i]; current_dwell = 1
        avg_dwell_times[current_state] += current_dwell; state_counts[current_state] += 1
        for state in range(num_states):
            if state_counts[state] > 0: avg_dwell_times[state] /= state_counts[state]
        return avg_dwell_times

    def calculate_transition_matrix(state_assignments, num_states):
        transitions = np.zeros((num_states, num_states))
        if len(state_assignments) < 2: return transitions.flatten()
        for i in range(len(state_assignments) - 1):
            from_state = state_assignments[i]; to_state = state_assignments[i+1]
            if from_state != to_state: transitions[from_state, to_state] += 1
        row_sums = transitions.sum(axis=1)
        normalized_transitions = np.divide(transitions, row_sums[:, np.newaxis], out=np.zeros_like(transitions), where=row_sums[:, np.newaxis] != 0)
        return normalized_transitions.flatten()

    # Import phenotype loader function (can be reused or copied)
    from preprocess_baseline import load_phenotype_data_for_fchnn # Assumes file exists

except ImportError as e:
    print(f"ERROR: Required library not found: {e}")
    print("Ensure 'connattractor' and dependencies are installed and preprocess_baseline.py exists if reusing functions.")
    exit()
except Exception as e_import:
     print(f"An error occurred during imports: {e_import}")
     exit()

def main():
    prep_start_time = time.time()
    # --- Load Configuration ---
    config_filename = 'config_custom_fchnn.yaml'
    print(f"Loading configuration from {config_filename}...")
    try:
        with open(config_filename, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"Error loading {config_filename}: {e}"); return

    data_cfg = config['data_params']
    fchnn_cfg = config['fchnn_feature_params']
    embedding_dir = data_cfg['embedding_output_dir']
    embedding_file = os.path.join(embedding_dir, 'custom_chnn_embedding.pkl')
    output_dir = data_cfg['feature_output_dir']
    region_dir = data_cfg['region_dir']
    num_regions = data_cfg['num_regions']
    sub_id_col = data_cfg['sub_id_col']
    target_col = data_cfg['target_col']

    # --- Load Custom fcHNN Embedding ---
    print(f"Loading custom embedding object from: {embedding_file}")
    try:
        with open(embedding_file, 'rb') as f:
            chnn_projection = pickle.load(f)
        # Verify loaded object
        if not (hasattr(chnn_projection, 'embedding_model') and hasattr(chnn_projection, 'attractor_coords')):
             raise ValueError("Loaded object does not appear to be a valid chnn_projection.")
        num_attractor_states = chnn_projection.attractor_coords.shape[0]
        attractor_coordinates = chnn_projection.attractor_coords[:, :2] # Use XY coords
        print(f"Successfully loaded embedding with {num_attractor_states} attractors.")
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {embedding_file}.")
        print("Please run build_custom_fchnn_embedding.py first.")
        return
    except Exception as e:
        print(f"Error loading embedding object: {e}"); return

    # --- Load and Preprocess Phenotype Data ---
    # Reusing the function - make sure it's available
    df_pheno_processed, final_pheno_feature_names, site_id_map, num_pheno_features = load_phenotype_data_for_fchnn(config)
    if df_pheno_processed is None: return

    # --- Process Subjects: Extract fcHNN Features using CUSTOM Embedding ---
    print("Processing subjects and extracting fcHNN dynamic features using CUSTOM embedding...")
    all_features_list = []
    all_targets = []
    processed_sub_ids = []
    ts_scaler = StandardScaler() # For normalizing time series

    for index, row in tqdm(df_pheno_processed.iterrows(), total=len(df_pheno_processed), desc="Extracting Features"):
        sub_id = None
        site_id = None
        try:
            sub_id = int(row[sub_id_col])
            site_id = site_id_map.get(sub_id)
            if site_id is None: continue

            filename = f"{site_id}_{sub_id:07d}_rois_cc200.1D"
            fpath = os.path.join(region_dir, filename)

            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0: continue

            ts_raw = np.loadtxt(fpath)
            if ts_raw.ndim != 2 or ts_raw.shape[1] != num_regions or ts_raw.shape[0] < 5: continue

            ts_normalized = ts_scaler.fit_transform(ts_raw)
            if np.isnan(ts_normalized).any(): ts_normalized = np.nan_to_num(ts_normalized)

            # --- Extract fcHNN Dynamic Features ---
            projected_coords = chnn_projection.embedding_model.transform(ts_normalized)
            if not np.all(np.isfinite(projected_coords)): continue

            state_assignments = assign_states_to_timepoints(projected_coords, attractor_coordinates)

            dynamic_features = []
            if fchnn_cfg.get('include_fractional_occupancy', True):
                dynamic_features.extend(calculate_fractional_occupancy(state_assignments, num_attractor_states))
            if fchnn_cfg.get('include_dwell_times', True):
                dynamic_features.extend(calculate_dwell_times(state_assignments, num_attractor_states))
            if fchnn_cfg.get('include_transition_matrix', True):
                dynamic_features.extend(calculate_transition_matrix(state_assignments, num_attractor_states))

            fchnn_dynamic_vector = np.array(dynamic_features, dtype=np.float32)
            if not np.all(np.isfinite(fchnn_dynamic_vector)): continue
            # ------------------------------------

            # Get Phenotype Vector
            if num_pheno_features > 0:
                 pheno_vector = row[final_pheno_feature_names].values.astype(np.float32)
                 if not np.all(np.isfinite(pheno_vector)): continue
            else: pheno_vector = np.array([], dtype=np.float32)

            # Combine
            final_feature_vector = np.concatenate((fchnn_dynamic_vector, pheno_vector))
            if not np.all(np.isfinite(final_feature_vector)): continue

            all_features_list.append(final_feature_vector)
            all_targets.append(int(row[target_col]))
            processed_sub_ids.append(sub_id)

        except FileNotFoundError: continue
        except Exception as e: print(f"Error processing subject {sub_id} (Index {index}): {e}"); continue

    # --- Save Processed Data ---
    if not all_features_list: print("Error: No features extracted."); return

    X = np.vstack(all_features_list)
    y = np.array(all_targets)
    subjects = np.array(processed_sub_ids)

    print(f"\nFinal processed feature matrix shape: {X.shape}")
    print(f"Final labels shape: {y.shape}")
    print(f"Number of subjects processed: {len(subjects)}")

    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')
    subjects_path = os.path.join(output_dir, 'subject_ids.npy')

    np.save(feature_path, X); np.save(label_path, y); np.save(subjects_path, subjects)
    print(f"\nSaved fcHNN dynamic features to: {output_dir}")
    prep_end_time = time.time()
    print(f"--- Custom fcHNN Preprocessing Finished ({prep_end_time - prep_start_time:.2f} seconds) ---")

if __name__ == "__main__":
    main()