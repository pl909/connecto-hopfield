import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm

# --- load_data remains the same ---
def load_data(config):
    """Loads phenotype data, constructs expected TS filenames, checks existence, and preprocesses features."""
    phenotype_file = config['data_params']['phenotype_file']
    sub_id_col = config['data_params']['sub_id_col']
    site_id_col = config['data_params'].get('site_id_col', 'SITE_ID') # Get site ID column name
    target_col = config['data_params']['target_col']
    num_cols_specified = config['data_params']['phenotype_cols_numerical']
    cat_cols_specified = config['data_params']['phenotype_cols_categorical']
    exclude_cols = config['data_params'].get('exclude_cols', [])
    region_dir = config['data_params']['region_dir']
    num_regions = config['data_params']['num_regions'] # Needed for validation later potentially

    print(f"Loading phenotype data from: {phenotype_file}")
    try:
        df_pheno_full = pd.read_csv(phenotype_file)
        if site_id_col not in df_pheno_full.columns:
             print(f"Error: SITE_ID column '{site_id_col}' not found in phenotype data.")
             return None, None, None
        # Create map BEFORE potential column removal by get_dummies
        site_id_map = df_pheno_full.set_index(sub_id_col)[site_id_col].to_dict()

    except FileNotFoundError:
        print(f"Error: Phenotype file not found at {phenotype_file}")
        return None, None, None
    except Exception as e:
        print(f"Error reading phenotype file {phenotype_file}: {e}")
        return None, None, None

    required_cols = [sub_id_col, target_col, site_id_col] + num_cols_specified + cat_cols_specified
    missing_cols = [col for col in required_cols if col not in df_pheno_full.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
        return None, None, None

    cols_to_keep = list(set([sub_id_col, target_col, site_id_col] + num_cols_specified + cat_cols_specified))
    df_pheno = df_pheno_full[cols_to_keep].copy()

    print(f"Original selected data shape: {df_pheno.shape}")
    print(f"Using target column: '{target_col}'")

    df_pheno = df_pheno.replace([-9999, -9999.0], np.nan)
    df_pheno[target_col] = df_pheno[target_col].apply(lambda x: 0 if x == 2 else 1 if x == 1 else -1)
    original_rows = len(df_pheno)
    df_pheno.dropna(subset=[target_col], inplace=True)
    df_pheno = df_pheno[df_pheno[target_col] != -1].copy()
    print(f"Filtered out {original_rows - len(df_pheno)} rows with invalid/missing target values.")
    print(f"Phenotype data shape after target filtering: {df_pheno.shape}")

    print("Processing specified numerical features...")
    processed_num_cols = []
    if num_cols_specified:
        valid_num_cols = [c for c in num_cols_specified if c in df_pheno.columns]
        if valid_num_cols:
            print(f" Processing numerical columns: {valid_num_cols}")
            num_imputer = SimpleImputer(strategy='mean')
            df_pheno[valid_num_cols] = num_imputer.fit_transform(df_pheno[valid_num_cols])
            scaler = StandardScaler()
            df_pheno[valid_num_cols] = scaler.fit_transform(df_pheno[valid_num_cols])
            processed_num_cols = valid_num_cols
        else: print("Warning: None of the specified numerical columns found.")
    else: print("No numerical columns specified in config.")

    print("Processing specified categorical features...")
    processed_cat_cols_encoded = []
    df_pheno_encoded = df_pheno.copy()
    if cat_cols_specified:
         valid_cat_cols = [c for c in cat_cols_specified if c in df_pheno.columns]
         if valid_cat_cols:
             print(f" Processing categorical columns: {valid_cat_cols}")
             cat_imputer = SimpleImputer(strategy='most_frequent')
             df_pheno[valid_cat_cols] = cat_imputer.fit_transform(df_pheno[valid_cat_cols])
             df_pheno_encoded = pd.get_dummies(df_pheno, columns=valid_cat_cols, dummy_na=False, dtype=float)
             processed_cat_cols_encoded = [c for c in df_pheno_encoded.columns if any(f"{cat_col}_" in c for cat_col in valid_cat_cols)]
             print(f"  Encoded categorical columns: {processed_cat_cols_encoded}")
         else: print("Warning: None of the specified categorical columns found.")
    else: print("No categorical columns specified in config.")

    final_feature_cols = processed_num_cols + processed_cat_cols_encoded
    num_pheno_features = len(final_feature_cols)

    if num_pheno_features > 0:
        print(f"Final phenotype features identified ({num_pheno_features}): {final_feature_cols}")
        valid_final_features = [c for c in final_feature_cols if c in df_pheno_encoded.columns]
        if valid_final_features:
             df_pheno_encoded[valid_final_features] = df_pheno_encoded[valid_final_features].fillna(0)
             final_feature_cols = valid_final_features
             num_pheno_features = len(final_feature_cols)
        else:
             print("Error: No valid final feature columns exist after processing.")
             num_pheno_features = 0
    else:
         print("No phenotype features specified or generated.")


    # --- Filter subjects based on TS file existence ---
    print(f"Checking for corresponding TS files in: {region_dir}")
    subject_data = {}
    subjects_processed = 0
    subjects_matched = 0
    subjects_missing_file = 0
    subjects_empty_file = 0

    # Iterate through the potentially encoded dataframe to get features later
    for index, row in tqdm(df_pheno_encoded.iterrows(), total=len(df_pheno_encoded), desc="Matching Files"):
        subjects_processed += 1
        sub_id = None
        site_id = None
        try:
            sub_id = int(row[sub_id_col])
            # Retrieve original SITE_ID using the map created earlier
            site_id = site_id_map.get(sub_id)

            if site_id is None:
                 subjects_missing_file += 1
                 continue

            # Construct filename - Pad SUB_ID with leading zeros to 7 digits
            filename = f"{site_id}_{sub_id:07d}_rois_cc200.1D"
            fpath = os.path.join(region_dir, filename)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                # Extract the final feature vector for this subject
                if num_pheno_features > 0 and valid_final_features:
                     feature_vector = row[valid_final_features].values.astype(np.float32)
                else:
                     feature_vector = np.array([], dtype=np.float32) # Still use empty if no features

                subject_data[sub_id] = {
                    'ts_path': fpath,
                    'pheno': feature_vector,
                    'target': int(row[target_col])
                }
                subjects_matched += 1
            elif not os.path.exists(fpath):
                subjects_missing_file += 1
            else:
                subjects_empty_file += 1

        except (ValueError, TypeError) as e:
            # print(f"Warning: Invalid SUB_ID {row[sub_id_col]} or SITE_ID {site_id} at index {index}: {e}")
            subjects_missing_file += 1
        except Exception as e_inner:
             print(f"Error processing row index {index}, SUB_ID {sub_id}: {e_inner}")
             subjects_missing_file += 1

    print(f"Processed {subjects_processed} subjects from phenotype data.")
    print(f"Found existing, non-empty TS files for {subjects_matched} subjects.")
    print(f"Skipped {subjects_missing_file} subjects (missing file or invalid ID/Site).")
    print(f"Skipped {subjects_empty_file} subjects (empty file).")

    if not subject_data:
        print("Error: No subjects remaining after matching phenotype and existing TS files.")
        return None, None, None

    subject_ids = list(subject_data.keys())
    all_targets_matched = np.array([subject_data[sid]['target'] for sid in subject_ids])

    return subject_data, num_pheno_features, all_targets_matched


# --- UPDATED FUNCTION ---
def load_and_pad_timeseries(subject_data, subject_ids_in_fold, max_len, num_regions):
    """Loads timeseries [Time, Region], normalizes, optionally truncates time, pads time."""
    timeseries_list = []
    actual_lengths = []
    print(f"Loading time series for {len(subject_ids_in_fold)} subjects in fold...")
    successful_loads = 0
    failed_loads = []
    ts_scaler = StandardScaler()

    # Determine padding target length based on max_len value
    truncate_active = (max_len > 0)
    if truncate_active:
        padding_target_len = max_len
        print(f"Truncation active. Max length set to: {max_len}")
    else:
        # Need to find max length in this specific batch if not truncating
        # This requires loading first, then finding max, then padding (two passes or store in memory)
        # For simplicity here, we load into memory first
        print("No truncation active. Will pad to max length found in this fold's successful loads.")
        # Placeholder, will calculate after loading
        padding_target_len = 0


    for sub_id in tqdm(subject_ids_in_fold, desc="Loading TS", leave=False):
        if sub_id not in subject_data:
            failed_loads.append(sub_id)
            continue

        fpath = subject_data[sub_id]['ts_path']
        try:
            ts_raw = np.loadtxt(fpath)

            if ts_raw.ndim != 2 or ts_raw.shape[1] != num_regions or ts_raw.shape[0] == 0:
                 failed_loads.append(sub_id)
                 continue

            ts_normalized = ts_scaler.fit_transform(ts_raw)
            if np.isnan(ts_normalized).any():
                 ts_normalized = np.nan_to_num(ts_normalized)

            ts = torch.tensor(ts_normalized, dtype=torch.float32)

            current_len = ts.shape[0]

            # --- Apply Truncation ONLY if max_len is positive ---
            if truncate_active and current_len > max_len:
                 ts = ts[:max_len, :] # Truncate rows
                 current_len = max_len # Update length
            # ----------------------------------------------------

            timeseries_list.append(ts) # Store potentially truncated tensor
            actual_lengths.append(current_len)
            successful_loads += 1
        except Exception as e:
            print(f"Error loading/processing {fpath} for subject {sub_id}: {e}")
            failed_loads.append(sub_id)

    if not timeseries_list:
        print("Error: No time series could be loaded successfully for this fold.")
        return None, None, None

    if failed_loads:
        print(f"Warning: Failed to load/process time series for {len(failed_loads)} subjects in this fold.")

    # --- Determine Final Padding Length ---
    if not truncate_active:
        # Find max length among successfully loaded (and potentially truncated) sequences
        if actual_lengths:
            padding_target_len = max(actual_lengths)
            print(f"Padding to max length found in fold: {padding_target_len}")
        else: # Should not happen if timeseries_list is not empty
             print("Error: No valid lengths found to determine padding target.")
             return None, None, None

    if padding_target_len <= 0:
        print(f"Error: Invalid padding target length ({padding_target_len}).")
        return None, None, None
    # -----------------------------------

    print(f"Padding {successful_loads} time series to final length: {padding_target_len}")

    # Pad sequences along the time dimension (dim 0)
    padded_individual = [F.pad(ts, (0, 0, 0, padding_target_len - ts.shape[0]), mode='constant', value=0.0) for ts in timeseries_list]
    padded_stacked = torch.stack(padded_individual, dim=0)
    padded_ts = padded_stacked.permute(1, 0, 2).contiguous() # [SeqLen, Batch, num_regions]

    # Create padding mask (True where padded) - shape [Batch, SeqLen]
    src_key_padding_mask = torch.ones(successful_loads, padding_target_len, dtype=torch.bool)
    for i, length in enumerate(actual_lengths):
        if length > 0:
            src_key_padding_mask[i, :length] = False # Valid steps are False

    # Return the actual padding length used, needed for PositionalEncoding
    return padded_ts, src_key_padding_mask, failed_loads, padding_target_len


# --- ADD THIS FUNCTION BACK (or keep similar logic) ---
def get_fold_max_len(subject_data, subject_ids_in_fold, num_regions):
    """Calculates max sequence length for subjects in the fold for Positional Encoding."""
    max_len_fold = 0
    print(f"Calculating max TS length for Positional Encoding ({len(subject_ids_in_fold)} subjects)...")
    count = 0
    valid_lengths = []
    for sub_id in tqdm(subject_ids_in_fold, desc="Checking TS Length", leave=False):
         if sub_id not in subject_data:
             continue
         fpath = subject_data[sub_id]['ts_path']
         try:
             # A more robust way to estimate length quickly might involve reading lines
             # For now, using loadtxt as before, assuming it's fast enough
             ts_raw = np.loadtxt(fpath)
             if ts_raw.ndim == 2 and ts_raw.shape[1] == num_regions and ts_raw.shape[0] > 0:
                 length = ts_raw.shape[0]
                 valid_lengths.append(length)
                 count += 1
             # else: skip invalid shapes
         except Exception as e:
             # print(f"Warning: Could not read/check {fpath} for subject {sub_id}: {e}")
             continue

    if not valid_lengths:
         print("Error: Could not determine max length from any file for Positional Encoding.")
         return None # Indicate failure

    max_len_fold = max(valid_lengths)
    print(f"Determined max TS length {max_len_fold} from {count} valid files for Positional Encoding.")
    return max_len_fold