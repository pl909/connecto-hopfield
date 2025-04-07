import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np

# Add src directory to Python path to allow importing utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

try:
    from src.utils import load_config, setup_logging, get_subject_id_from_path
except ImportError as e:
    print(f"Error importing modules: {e}. Make sure you are running from the project root or have the 'src' directory in your PYTHONPATH.")
    sys.exit(1)

def run_qc(config_path):
    """Loads data, applies QC filters, and saves included subjects list."""
    config = load_config(config_path)

    # --- Setup Logging --- (Log file placed in the base results directory)
    results_base_dir = config['paths']['results_base_dir']
    os.makedirs(results_base_dir, exist_ok=True)
    log_file = os.path.join(results_base_dir, 'qc_filter.log')
    setup_logging(log_file)
    logging.info(f"Running QC Filtering with config: {config_path}")

    # --- Define Paths --- #
    base_dir = config['paths']['base_dir']
    
    # Use paths directly from config if available, otherwise construct them
    fd_file = config['qc'].get('fd_summary_file', os.path.join(base_dir, 'func_preproc', 'popFD_max.txt'))
    scrub_file = config['qc'].get('percent_scrubbed_file', os.path.join(base_dir, 'func_preproc', 'pop_percent_scrubbed.txt'))
    output_subject_list_file = os.path.join(results_base_dir, 'included_subjects.txt')

    # --- Load QC Data --- #
    try:
        logging.info(f"Loading FD data from: {fd_file}")
        # Adjust separator and header based on actual file format
        fd_data = pd.read_csv(fd_file, sep='\s+', header=None, names=['subject_run', 'max_fd'])
        logging.info(f"Loaded {len(fd_data)} FD entries.")

        logging.info(f"Loading Scrubbing data from: {scrub_file}")
        # Adjust separator and header based on actual file format
        scrub_data = pd.read_csv(scrub_file, sep='\s+', header=None, names=['subject_run', 'percent_scrubbed'])
        logging.info(f"Loaded {len(scrub_data)} scrubbing entries.")

    except FileNotFoundError as e:
        logging.error(f"QC file not found: {e}. Please ensure the directory exists at {base_dir} and contains the required QC files.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading QC data: {e}")
        sys.exit(1)

    # --- Extract Subject IDs --- #
    # Assuming format like 'sub-01_task-pain_run-01'
    fd_data['subject_id'] = fd_data['subject_run'].apply(lambda x: get_subject_id_from_path(x) or 'parse_error')
    scrub_data['subject_id'] = scrub_data['subject_run'].apply(lambda x: get_subject_id_from_path(x) or 'parse_error')

    # Filter out any parsing errors
    fd_data = fd_data[fd_data['subject_id'] != 'parse_error']
    scrub_data = scrub_data[scrub_data['subject_id'] != 'parse_error']

    if fd_data.empty or scrub_data.empty:
        logging.error("Could not extract valid subject IDs from QC file paths. Check file naming conventions.")
        sys.exit(1)

    # --- Aggregate QC per Subject (take the max across runs) --- #
    logging.info("Aggregating QC metrics per subject (taking max FD and max scrub percentage across runs).")
    subject_fd_max = fd_data.groupby('subject_id')['max_fd'].max().reset_index()
    subject_scrub_max = scrub_data.groupby('subject_id')['percent_scrubbed'].max().reset_index()

    # --- Merge QC Data --- #
    qc_merged = pd.merge(subject_fd_max, subject_scrub_max, on='subject_id', how='outer')
    # Fill NaN for subjects present in one file but not the other (might indicate issues)
    qc_merged = qc_merged.fillna({'max_fd': np.inf, 'percent_scrubbed': np.inf})
    initial_subjects = qc_merged['subject_id'].unique()
    logging.info(f"Found {len(initial_subjects)} unique subjects in QC files.")

    # --- Apply QC Filters --- #
    max_fd_thresh = config['qc'].get('max_fd', 5.0)
    max_scrub_thresh = config['qc'].get('max_scrub', 25.0)
    logging.info(f"Applying QC filters: max_fd <= {max_fd_thresh}, max_scrub <= {max_scrub_thresh}")

    fd_passed = qc_merged['max_fd'] <= max_fd_thresh
    scrub_passed = qc_merged['percent_scrubbed'] <= max_scrub_thresh
    qc_passed = fd_passed & scrub_passed

    included_subjects_df = qc_merged[qc_passed]
    excluded_subjects_df = qc_merged[~qc_passed]

    logging.info(f"QC Results: {len(included_subjects_df)} subjects passed, {len(excluded_subjects_df)} subjects excluded.")
    if not excluded_subjects_df.empty:
        logging.info("Excluded subjects and reasons:")
        for _, row in excluded_subjects_df.iterrows():
            reasons = []
            if not (row['max_fd'] <= max_fd_thresh):
                reasons.append(f"max_fd={row['max_fd']:.2f}")
            if not (row['percent_scrubbed'] <= max_scrub_thresh):
                reasons.append(f"percent_scrubbed={row['percent_scrubbed']:.1f}%")
            logging.info(f"  - {row['subject_id']}: {'; '.join(reasons)}")

    included_subject_ids = included_subjects_df['subject_id'].tolist()

    if not included_subject_ids:
        logging.error("No subjects passed QC filters or no subjects found. Cannot proceed.")
        sys.exit(1)

    # --- Save Included Subjects --- #
    try:
        with open(output_subject_list_file, 'w') as f:
            for sub_id in sorted(included_subject_ids):
                f.write(f"{sub_id}\n")
        logging.info(f"List of {len(included_subject_ids)} included subjects saved to: {output_subject_list_file}")
    except Exception as e:
        logging.error(f"Error saving included subjects list: {e}")
        sys.exit(1)

    logging.info("QC Filtering finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quality Control Filtering based on RPN outputs.")
    parser.add_argument("config", help="Path to the configuration YAML file (e.g., configs/achnn_config.yaml)")
    args = parser.parse_args()

    run_qc(args.config) 