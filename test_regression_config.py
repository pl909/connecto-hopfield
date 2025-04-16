#!/usr/bin/env python
"""
Test script to validate the updated regression configuration.
This loads the config and prints key settings related to subject ID handling.
"""
import os
import sys
import logging
import yaml
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Test the regression configuration file"""
    logging.info("Loading regression config...")
    config_path = 'configs/achnn_regression_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print the subject ID and file ID columns
    subject_id_col = config.get('data', {}).get('subject_id_col')
    file_id_col = config.get('data', {}).get('file_id_col')
    regression_target = config.get('data', {}).get('regression_target')
    
    logging.info(f"Subject ID column: {subject_id_col}")
    logging.info(f"File ID column: {file_id_col}")
    logging.info(f"Regression target: {regression_target}")
    
    # Check if the clinical data exists
    clinical_data_path = config.get('data', {}).get('clinical_data_path')
    if not clinical_data_path:
        clinical_data_path = os.path.join(
            config.get('paths', {}).get('base_dir', ''),
            config.get('paths', {}).get('phenotypic_file', '')
        )
    
    logging.info(f"Clinical data path: {clinical_data_path}")
    
    if os.path.exists(clinical_data_path):
        logging.info("✅ Clinical data file exists")
        try:
            clinical_df = pd.read_csv(clinical_data_path)
            logging.info(f"Clinical data shape: {clinical_df.shape}")
            
            # Check if subject_id_col exists
            if subject_id_col in clinical_df.columns:
                logging.info(f"✅ Subject ID column '{subject_id_col}' exists in clinical data")
                logging.info(f"Sample subject IDs: {clinical_df[subject_id_col].head().tolist()}")
            else:
                logging.error(f"❌ Subject ID column '{subject_id_col}' NOT found in clinical data")
                logging.info(f"Available columns: {clinical_df.columns.tolist()}")
            
            # Check if file_id_col exists
            if file_id_col in clinical_df.columns:
                logging.info(f"✅ File ID column '{file_id_col}' exists in clinical data")
                logging.info(f"Sample file IDs: {clinical_df[file_id_col].head().tolist()}")
            else:
                logging.error(f"❌ File ID column '{file_id_col}' NOT found in clinical data")
            
            # Check if regression_target exists
            if regression_target in clinical_df.columns:
                logging.info(f"✅ Regression target '{regression_target}' exists in clinical data")
                target_stats = clinical_df[regression_target].describe()
                logging.info(f"Target statistics: min={target_stats['min']}, max={target_stats['max']}, "
                           f"mean={target_stats['mean']:.2f}, count={target_stats['count']}")
            else:
                logging.error(f"❌ Regression target '{regression_target}' NOT found in clinical data")
                
        except Exception as e:
            logging.error(f"❌ Error reading clinical data: {e}")
    else:
        logging.error(f"❌ Clinical data file not found at: {clinical_data_path}")
    
    # Check timeseries directory
    timeseries_dir = config.get('data', {}).get('timeseries_dir')
    if not timeseries_dir:
        timeseries_dir = config.get('paths', {}).get('regional_timeseries_dir')
    
    logging.info(f"Timeseries directory: {timeseries_dir}")
    
    if os.path.exists(timeseries_dir):
        logging.info("✅ Timeseries directory exists")
        # List a few files in the directory
        sample_files = [f for f in os.listdir(timeseries_dir)[:5] if f.endswith('.1D')]
        logging.info(f"Sample timeseries files: {sample_files}")
    else:
        logging.error(f"❌ Timeseries directory not found at: {timeseries_dir}")
    
    logging.info("Config test completed")

if __name__ == "__main__":
    main() 