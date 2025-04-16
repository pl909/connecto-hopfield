#!/usr/bin/env python
import yaml
import logging
import sys
import pandas as pd
from src.data_loader import AbideTimeseriesProcessor, load_timeseries_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Load configuration
with open('configs/achnn_regression_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test single subject loading
subject_id = '50003'  # A known subject from ABIDE dataset
logging.info(f"Testing timeseries loading for subject {subject_id}...")
processor = AbideTimeseriesProcessor(config, subject_id)
timeseries = processor.process()
if timeseries is not None:
    logging.info(f"✅ Successfully loaded timeseries for subject {subject_id}, shape: {timeseries.shape}")
else:
    logging.error(f"❌ Failed to load timeseries for subject {subject_id}")

# Check phenotypic data directly
logging.info("Checking phenotypic data...")
phenotypic_file = 'Phenotypic_V1_0b_preprocessed1.csv'
try:
    pheno_df = pd.read_csv(phenotypic_file)
    logging.info(f"Loaded phenotypic data, shape: {pheno_df.shape}")
    
    # Check subject ID column
    subject_cols = ['subject', 'Subject', 'SUB_ID', 'FILE_ID']
    for col in subject_cols:
        if col in pheno_df.columns:
            logging.info(f"Found column '{col}' with sample values: {pheno_df[col].iloc[:5].tolist()}")
    
    # Check if subject_id is in the dataframe
    for col in subject_cols:
        if col in pheno_df.columns:
            found = subject_id in pheno_df[col].astype(str).values
            logging.info(f"Subject {subject_id} found in column '{col}': {found}")
            
            # Check numeric version too
            numeric_id = int(subject_id)
            found_numeric = numeric_id in pheno_df[col].astype(float).values
            logging.info(f"Subject {numeric_id} (numeric) found in column '{col}': {found_numeric}")
    
    # Try testing with a different subject format
    if 'FILE_ID' in pheno_df.columns:
        file_id_with_site = f"Pitt_{subject_id}"
        found_with_site = pheno_df['FILE_ID'].str.contains(file_id_with_site).any()
        logging.info(f"Subject '{file_id_with_site}' found in FILE_ID column: {found_with_site}")
        
        if found_with_site:
            matching_rows = pheno_df[pheno_df['FILE_ID'].str.contains(file_id_with_site)]
            logging.info(f"Found {len(matching_rows)} rows with FILE_ID matching '{file_id_with_site}'")
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                logging.info(f"First matching row - FILE_ID: {row['FILE_ID']}, subject: {row['subject'] if 'subject' in row else 'N/A'}")
    
except Exception as e:
    logging.error(f"❌ Error checking phenotypic data: {e}")

# Now try the full data loading with the correct subject format
logging.info("Testing load_timeseries_data function with corrected subject ID...")
try:
    # Get the correct subject ID format from the pheno data
    if 'subject' in pheno_df.columns:
        # Find the internal subject ID that matches our file_id
        if 'FILE_ID' in pheno_df.columns:
            matching_rows = pheno_df[pheno_df['FILE_ID'].str.contains(f"Pitt_{subject_id}")]
            if len(matching_rows) > 0:
                internal_subject_id = matching_rows.iloc[0]['subject']
                logging.info(f"Using internal subject ID: {internal_subject_id}")
                
                # Now try loading with the internal subject ID
                subject_data_dict, clinical_data = load_timeseries_data(
                    config, subject_ids=[internal_subject_id], task_type='regression'
                )
                if internal_subject_id in subject_data_dict:
                    windows = subject_data_dict[internal_subject_id]
                    logging.info(f"✅ Successfully loaded windows for subject {internal_subject_id}, shape: {windows.shape}")
                else:
                    logging.error(f"❌ Failed to load windows for subject {internal_subject_id}")
except Exception as e:
    logging.error(f"❌ Error in load_timeseries_data: {e}") 