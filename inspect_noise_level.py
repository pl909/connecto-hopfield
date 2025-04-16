import os
import glob
import pandas as pd
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Function largely borrowed from data_loader.py, simplified for inspection
def get_subject_files(config):
    """Finds subjects with existing, valid time series files."""
    phenotype_file = config['data_params']['phenotype_file']
    sub_id_col = config['data_params']['sub_id_col']
    site_id_col = config['data_params'].get('site_id_col', 'SITE_ID')
    region_dir = config['data_params']['region_dir']
    num_regions = config['data_params']['num_regions']

    print(f"Loading phenotype data from: {phenotype_file}")
    try:
        df_pheno_full = pd.read_csv(phenotype_file)
        if site_id_col not in df_pheno_full.columns or sub_id_col not in df_pheno_full.columns:
             print(f"Error: SUB_ID ('{sub_id_col}') or SITE_ID ('{site_id_col}') column not found.")
             return None
        site_id_map = df_pheno_full.set_index(sub_id_col)[site_id_col].to_dict()
    except Exception as e:
        print(f"Error reading phenotype file {phenotype_file}: {e}")
        return None

    print(f"Checking for corresponding TS files in: {region_dir}")
    subject_files = {}
    subjects_processed = 0
    subjects_matched_file_exists = 0
    subjects_valid_file = 0

    for index, row in tqdm(df_pheno_full.iterrows(), total=len(df_pheno_full), desc="Checking Files"):
        subjects_processed += 1
        sub_id = None
        site_id = None
        try:
            sub_id = int(row[sub_id_col])
            site_id = site_id_map.get(sub_id)
            if site_id is None: continue

            filename = f"{site_id}_{sub_id:07d}_rois_cc200.1D"
            fpath = os.path.join(region_dir, filename)

            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                subjects_matched_file_exists += 1
                # Quick check for basic validity (can we load it?)
                try:
                    # Try loading just the first line to check columns quickly
                    ts_test = np.loadtxt(fpath, max_rows=1)
                    # Check if it's a 1D array (implies single line) or 2D with correct columns
                    if (ts_test.ndim == 1 and len(ts_test) == num_regions) or \
                       (ts_test.ndim == 2 and ts_test.shape[1] == num_regions):
                        subject_files[sub_id] = fpath # Store path if valid structure
                        subjects_valid_file += 1
                    # else:
                    #    print(f"Warning: File {filename} has unexpected structure/columns. Skipping.")
                except Exception:
                     # print(f"Warning: Could not load/validate file {filename}. Skipping.")
                     pass # Ignore files that cause loading errors
            # else: File doesn't exist or is empty

        except (ValueError, TypeError, KeyError):
            continue # Skip rows with invalid IDs

    print(f"Processed {subjects_processed} subjects from phenotype data.")
    print(f"Found {subjects_matched_file_exists} existing, non-empty TS files.")
    print(f"Found {subjects_valid_file} files with expected region count ({num_regions}).")

    if not subject_files:
        print("Error: No valid subject time series files found.")
        return None

    return subject_files

# Main inspection logic
def inspect_signal_scale(config, subject_files):
    """Loads, normalizes, and analyzes std dev of time series."""
    num_regions = config['data_params']['num_regions']
    all_std_devs = []
    failed_loads = []
    scaler = StandardScaler()

    print(f"\nAnalyzing signal standard deviation for {len(subject_files)} subjects...")
    for sub_id, fpath in tqdm(subject_files.items(), desc="Analyzing TS"):
        try:
            ts_raw = np.loadtxt(fpath)
            if ts_raw.ndim != 2 or ts_raw.shape[1] != num_regions or ts_raw.shape[0] == 0:
                failed_loads.append(sub_id)
                continue

            # Apply Z-score normalization (same as in data_loader)
            ts_normalized = scaler.fit_transform(ts_raw)
            if np.isnan(ts_normalized).any():
                 # print(f"Warning: NaNs after scaling subject {sub_id}. Skipping std dev calculation.")
                 failed_loads.append(sub_id)
                 continue

            # Calculate overall standard deviation of the normalized signal matrix
            overall_std = np.std(ts_normalized)
            all_std_devs.append(overall_std)

        except Exception as e:
            print(f"Error processing file {fpath} for subject {sub_id}: {e}")
            failed_loads.append(sub_id)

    if failed_loads:
        print(f"\nWarning: Failed to process data for {len(failed_loads)} subjects.")

    if not all_std_devs:
        print("Error: Could not calculate standard deviations for any subject.")
        return

    all_std_devs = np.array(all_std_devs)

    print("\n--- Statistics of Normalized Time Series Standard Deviations ---")
    print(f"Number of subjects analyzed: {len(all_std_devs)}")
    print(f"Mean Std Dev:   {np.mean(all_std_devs):.4f}")
    print(f"Median Std Dev: {np.median(all_std_devs):.4f}")
    print(f"Min Std Dev:    {np.min(all_std_devs):.4f}")
    print(f"Max Std Dev:    {np.max(all_std_devs):.4f}")
    print(f"Std Dev of Std Devs: {np.std(all_std_devs):.4f}")

    # --- Suggestions for Noise Level ---
    mean_std = np.mean(all_std_devs)
    print("\n--- Suggested Noise Levels (Gaussian std dev) ---")
    print(f"Very Low ( ~1% of mean signal std): {mean_std * 0.01:.4f}")
    print(f"Low      ( ~5% of mean signal std): {mean_std * 0.05:.4f}")
    print(f"Moderate (~10% of mean signal std): {mean_std * 0.10:.4f}")
    print(f"High     (~20% of mean signal std): {mean_std * 0.20:.4f}")
    print("\nNote: Start with 'Low' or 'Moderate'. Too high can destroy signal, too low has no effect.")

    # --- Plot Histogram ---
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(all_std_devs, kde=True, bins=30)
        plt.title('Distribution of Subject-wise Normalized Time Series Standard Deviations')
        plt.xlabel('Standard Deviation of Normalized Signal')
        plt.ylabel('Number of Subjects')
        plt.grid(True, axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig('normalized_ts_std_dev_distribution.png')
        print("\nSaved histogram to 'normalized_ts_std_dev_distribution.png'")
        # plt.show() # Uncomment to display plot immediately if running interactively
    except Exception as plot_e:
        print(f"\nWarning: Could not generate histogram plot: {plot_e}")

if __name__ == "__main__":
    print("Loading configuration from config.yaml...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit()

    # Find valid subject files
    subject_files_dict = get_subject_files(config)

    # Analyze signal scale if files were found
    if subject_files_dict:
        inspect_signal_scale(config, subject_files_dict)
    else:
        print("Exiting: No valid subject files found based on phenotype list and file checks.")