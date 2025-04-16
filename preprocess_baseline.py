import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm

def calculate_fc_vector(timeseries_data):
    """Calculates the flattened upper triangle of the Pearson correlation matrix."""
    if not isinstance(timeseries_data, np.ndarray):
        timeseries_data = np.array(timeseries_data)

    # Ensure data is numeric and finite before correlation
    if not np.issubdtype(timeseries_data.dtype, np.number):
        print("Warning: Time series data is not numeric. Cannot calculate FC.")
        return None
    if not np.all(np.isfinite(timeseries_data)):
        # print("Warning: Non-finite values found in time series. Replacing with 0 before correlation.")
        timeseries_data = np.nan_to_num(timeseries_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate correlation, handle potential NaNs from zero variance
    try:
         correlation_matrix = np.corrcoef(timeseries_data.T) # Transpose to [Regions, Time]
         # Replace any NaNs that occurred immediately
         correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0) # Replace NaN with 0
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

    # Final check for safety
    if not np.all(np.isfinite(correlation_matrix)):
         print("Warning: Non-finite values persisted in correlation matrix. Returning None.")
         return None

    num_regions = correlation_matrix.shape[0]
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
         print(f"Error: Correlation matrix is not square ({correlation_matrix.shape}).")
         return None

    upper_triangle_indices = np.triu_indices(num_regions, k=1)
    fc_vector = correlation_matrix[upper_triangle_indices]
    return fc_vector

def main():
    # --- Load Configuration ---
    print("Loading configuration from config_baseline.yaml...")
    try:
        with open('config_baseline.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config_baseline.yaml: {e}")
        return

    data_cfg = config['data_params']
    phenotype_file = data_cfg['phenotype_file']
    sub_id_col = data_cfg['sub_id_col']
    site_id_col = data_cfg.get('site_id_col', 'SITE_ID')
    target_col = data_cfg['target_col']
    num_cols_specified = data_cfg['phenotype_cols_numerical']
    cat_cols_specified = data_cfg['phenotype_cols_categorical']
    region_dir = data_cfg['region_dir']
    num_regions = data_cfg['num_regions']
    output_dir = data_cfg['output_feature_dir']

    # --- Load Phenotype Data ---
    print(f"Loading phenotype data from: {phenotype_file}")
    try:
        df_pheno_full = pd.read_csv(phenotype_file)
        cols_needed_from_csv = list(set([sub_id_col, site_id_col, target_col] + num_cols_specified + cat_cols_specified))
        missing_cols = [c for c in cols_needed_from_csv if c not in df_pheno_full.columns]
        if missing_cols:
            print(f"Error: Missing required columns in CSV: {missing_cols}")
            return
        df_pheno = df_pheno_full[cols_needed_from_csv].copy()
    except Exception as e:
        print(f"Error reading phenotype file: {e}")
        return

    # --- Preprocess Phenotypes ---
    print("Preprocessing phenotype features...")
    df_pheno = df_pheno.replace([-9999, -9999.0, ''], np.nan) # Also treat empty strings as NaN

    # Convert relevant columns to numeric *before* imputation, coercing errors
    print("Attempting numeric conversion for specified columns...")
    for col in num_cols_specified + [target_col]: # Target should also be numeric
        if col in df_pheno.columns:
            df_pheno[col] = pd.to_numeric(df_pheno[col], errors='coerce')

    # Target processing (dropna must happen after numeric conversion)
    df_pheno[target_col] = df_pheno[target_col].apply(lambda x: 0 if x == 2 else 1 if x == 1 else -1)
    original_rows = len(df_pheno)
    df_pheno.dropna(subset=[target_col], inplace=True) # Drop rows where target became NaN or was NaN
    df_pheno = df_pheno[df_pheno[target_col] != -1].copy() # Filter invalid target maps (-1)
    print(f"Filtered {original_rows - len(df_pheno)} rows with invalid/missing target values.")
    print(f"Phenotype data shape after target filtering: {df_pheno.shape}")

    # Numerical features
    valid_num_cols = [c for c in num_cols_specified if c in df_pheno.columns]
    if valid_num_cols:
        print(f" Processing numerical columns: {valid_num_cols}")
        num_imputer = SimpleImputer(strategy='mean')
        df_pheno[valid_num_cols] = num_imputer.fit_transform(df_pheno[valid_num_cols])
        pheno_num_scaler = StandardScaler()
        df_pheno[valid_num_cols] = pheno_num_scaler.fit_transform(df_pheno[valid_num_cols])
    else: pheno_num_scaler = None

    # Categorical features
    valid_cat_cols = [c for c in cat_cols_specified if c in df_pheno.columns]
    if valid_cat_cols:
        print(f" Processing categorical columns: {valid_cat_cols}")
        # Impute first (convert to string to ensure consistent type for imputer/encoder)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_pheno[valid_cat_cols] = df_pheno[valid_cat_cols].astype(str) # Ensure string type
        df_pheno[valid_cat_cols] = cat_imputer.fit_transform(df_pheno[valid_cat_cols])
        # Fit OneHotEncoder
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df_pheno[valid_cat_cols])
        ohe_feature_names = ohe.get_feature_names_out(valid_cat_cols)
    else:
        ohe = None
        ohe_feature_names = []

    # --- Process Subjects and Extract Features ---
    print(f"Processing subjects and extracting FC features...")
    all_features = []
    all_targets = []
    processed_sub_ids = []
    ts_scaler = StandardScaler()

    for index, row in tqdm(df_pheno.iterrows(), total=len(df_pheno), desc="Extracting Features"):
        sub_id = None
        site_id = None
        try:
            sub_id = int(row[sub_id_col])
            site_id = row[site_id_col] # Get site ID for filename

            filename = f"{site_id}_{sub_id:07d}_rois_cc200.1D"
            fpath = os.path.join(region_dir, filename)

            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0: continue

            # Load Time Series
            ts_raw = np.loadtxt(fpath)
            if ts_raw.ndim != 2 or ts_raw.shape[1] != num_regions or ts_raw.shape[0] < 2: continue

            # Normalize Time Series
            ts_normalized = ts_scaler.fit_transform(ts_raw)
            if np.isnan(ts_normalized).any(): ts_normalized = np.nan_to_num(ts_normalized)

            # Calculate FC Vector
            fc_vector = calculate_fc_vector(ts_normalized)
            if fc_vector is None or not np.all(np.isfinite(fc_vector)):
                 # print(f"Skipping {sub_id}: FC calculation failed or non-finite.")
                 continue

            # Prepare Phenotype Vector for this subject
            pheno_num_part = np.array([], dtype=np.float32)
            if valid_num_cols:
                 # Extract numerical data, ensure it's float
                 pheno_num_part = row[valid_num_cols].values.astype(np.float32)
                 # Explicitly check for NaNs/Infs *before* concatenation
                 if not np.all(np.isfinite(pheno_num_part)):
                      print(f"Skipping {sub_id}: Non-finite values found in numerical phenotype features: {row[valid_num_cols].values}")
                      continue


            pheno_cat_part_encoded = np.array([], dtype=np.float32)
            if valid_cat_cols and ohe is not None:
                 # *** Use DataFrame slice for transform ***
                 cat_data_sample_df = pd.DataFrame([row[valid_cat_cols].astype(str)], columns=valid_cat_cols) # Ensure string type matches fit
                 pheno_cat_part_encoded = ohe.transform(cat_data_sample_df).flatten().astype(np.float32)
                 if not np.all(np.isfinite(pheno_cat_part_encoded)):
                      print(f"Skipping {sub_id}: Non-finite values found after one-hot encoding.")
                      continue

            # Concatenate features
            combined_pheno_vector = np.concatenate((pheno_num_part, pheno_cat_part_encoded))

            # Final check for NaNs in the combined phenotype vector
            # Catch TypeError here if non-numeric data slipped through somehow
            try:
                if np.isnan(combined_pheno_vector).any():
                    print(f"Skipping {sub_id}: NaNs found in final combined phenotype vector.")
                    continue
            except TypeError as e_nan_check:
                 print(f"Skipping {sub_id}: TypeError during final NaN check on phenotype vector. Data: {combined_pheno_vector}. Error: {e_nan_check}")
                 continue


            final_feature_vector = np.concatenate((fc_vector, combined_pheno_vector))

            # Final check on the whole vector
            if not np.all(np.isfinite(final_feature_vector)):
                 print(f"Skipping {sub_id}: Non-finite values found in final combined feature vector.")
                 continue

            all_features.append(final_feature_vector)
            all_targets.append(int(row[target_col]))
            processed_sub_ids.append(sub_id)

        except FileNotFoundError: continue
        except ValueError as ve: # Catch potential errors during int conversion or other processing
             print(f"ValueError processing subject {sub_id} (Index {index}): {ve}")
             continue
        except Exception as e:
            print(f"Unexpected error processing subject {sub_id} (Index {index}): {e}")
            continue # Skip subject on any other error

    # --- Save Processed Data ---
    if not all_features:
        print("Error: No features were successfully extracted for any subject.")
        return

    X = np.vstack(all_features)
    y = np.array(all_targets)
    subjects = np.array(processed_sub_ids)

    print(f"\nFinal processed feature matrix shape: {X.shape}")
    print(f"Final labels shape: {y.shape}")
    print(f"Number of subjects with processed features: {len(subjects)}")

    os.makedirs(output_dir, exist_ok=True)
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')
    subjects_path = os.path.join(output_dir, 'subject_ids.npy')

    np.save(feature_path, X)
    np.save(label_path, y)
    np.save(subjects_path, subjects)

    print(f"\nSaved features to: {feature_path}")
    print(f"Saved labels to: {label_path}")
    print(f"Saved subject IDs to: {subjects_path}")
    print("--- Preprocessing Finished ---")

if __name__ == "__main__":
    main()