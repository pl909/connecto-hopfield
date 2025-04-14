#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify access to required data files for ABIDE ACHNN project.
Checks for:
- ABIDE Phenotypic CSV file
- Preprocessed ROI Timeseries files (.1D format)
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
import glob
from datetime import datetime

# Add src directory to Python path to allow importing utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)

def setup_logging():
    """Set up logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def check_phenotypic_file(base_dir, config):
    """Checks if the main phenotypic CSV file is accessible."""
    pheno_path = os.path.join(base_dir, config['paths']['phenotypic_file'])
    if not os.path.exists(pheno_path):
        logging.error(f"Phenotypic CSV file not found: {pheno_path}")
        return False
    
    logging.info(f"Phenotypic CSV file exists: {pheno_path}")
    try:
        df = pd.read_csv(pheno_path)
        logging.info(f"  Successfully read phenotypic data with shape: {df.shape}")
        # Check for essential columns
        required_cols = ['SUB_ID', 'SITE_ID', 'FILE_ID', 'DX_GROUP']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"  Missing one or more required columns in CSV: {required_cols}")
            return False
        logging.info(f"  Found required columns: {required_cols}")
        return True
    except Exception as e:
        logging.error(f"  Error reading phenotypic CSV file: {e}")
        return False

def check_regional_timeseries(config):
    """Check if regional timeseries files (.1D) are accessible."""
    regional_ts_dir = config['paths']['regional_timeseries_dir']
    
    if not os.path.exists(regional_ts_dir):
        logging.error(f"Regional timeseries directory not found: {regional_ts_dir}")
        return False
    
    logging.info(f"Regional timeseries directory exists: {regional_ts_dir}")
    
    # Check for .1D files (adjust pattern if derivative name changes)
    derivative_name = os.path.basename(regional_ts_dir).split('_')[-1]
    ts_pattern = os.path.join(regional_ts_dir, f"*_{derivative_name}.1D")
    ts_files = glob.glob(ts_pattern)
    
    if not ts_files:
        logging.error(f"No timeseries files found in {regional_ts_dir} with pattern '*{derivative_name}.1D'")
        return False
    
    logging.info(f"Found {len(ts_files)} timeseries files. Examples:")
    for i, ts_file in enumerate(ts_files[:5]):
        logging.info(f"  {i+1}. {os.path.basename(ts_file)}")
    
    # Try to read one file to check format
    try:
        test_file = ts_files[0]
        logging.info(f"Testing file read: {test_file}")
        # ABIDE .1D files are often whitespace delimited, sometimes without header
        ts_data = pd.read_csv(test_file, delim_whitespace=True, header=None)
        expected_regions = config['data']['num_regions']
        if ts_data.shape[1] != expected_regions:
             logging.warning(f"  Read data shape {ts_data.shape} - Mismatch with expected regions ({expected_regions}). Check atlas/config.")
        else:
             logging.info(f"  Successfully read data with shape: {ts_data.shape}")
        return True
    except pd.errors.EmptyDataError:
        logging.error(f"Timeseries file is empty: {test_file}")
        return False
    except Exception as e:
        logging.error(f"Error reading timeseries file {test_file}: {e}")
        return False

def main(config_path):
    """Main function to test data access for ABIDE."""
    setup_logging()
    logging.info(f"Testing ABIDE data access with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Get base directory
    base_dir = config['paths']['base_dir']
    logging.info(f"Base directory (for phenotypic file): {base_dir}")
    
    if not os.path.exists(base_dir):
        logging.error(f"Base directory not found: {base_dir}")
        sys.exit(1)
    
    overall_success = True
    
    # Check Phenotypic file
    logging.info("\n--- Testing Phenotypic CSV Access ---")
    pheno_success = check_phenotypic_file(base_dir, config)
    overall_success = overall_success and pheno_success

    # Check regional timeseries
    logging.info("\n--- Testing Regional Timeseries Access ---")
    ts_success = check_regional_timeseries(config)
    overall_success = overall_success and ts_success
    
    # Summary
    logging.info("\n--- Summary ---")
    if overall_success:
        logging.info("✅ All essential ABIDE data access tests PASSED.")
    else:
        logging.warning("❌ Some essential ABIDE data access tests FAILED. Review logs.")
    
    return overall_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data access for ABIDE ACHNN configuration.")
    parser.add_argument("config", help="Path to the configuration YAML file (e.g., configs/achnn_config_abide.yaml)")
    args = parser.parse_args()
    
    main(args.config)