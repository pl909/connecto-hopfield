#!/usr/bin/env python
"""
Simple test script for the regression pipeline.
This loads data for a few sample subjects and prints the results.
"""
import os
import sys
import logging
import yaml
import numpy as np
from src.data_loader import load_timeseries_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Test the regression pipeline with a few sample subjects"""
    logging.info("Loading config...")
    config_path = 'configs/achnn_regression_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with a few known subject IDs
    test_subject_ids = ['50003', '50004', '50005']
    logging.info(f"Testing data loading with subjects: {test_subject_ids}")
    
    try:
        # Load the data for these subjects
        subject_data_dict, clinical_data = load_timeseries_data(
            config, subject_ids=test_subject_ids, task_type='regression'
        )
        
        logging.info(f"Successfully loaded data for {len(subject_data_dict)} subjects")
        
        # Print info about each subject's data
        for subject_id, windows in subject_data_dict.items():
            logging.info(f"Subject {subject_id}: {len(windows)} windows, shape: {windows.shape}")
            
            # Show ADOS_TOTAL score for this subject if available
            regression_target = config.get('data', {}).get('regression_target', 'ADOS_TOTAL')
            subject_id_col = config.get('data', {}).get('subject_id_col', 'subject')
            
            if regression_target in clinical_data.columns:
                subject_rows = clinical_data[clinical_data[subject_id_col] == subject_id]
                if len(subject_rows) > 0:
                    target_value = subject_rows[regression_target].iloc[0]
                    logging.info(f"Subject {subject_id} {regression_target}: {target_value}")
        
        # All done!
        if len(subject_data_dict) > 0:
            logging.info("✅ Test completed successfully!")
        else:
            logging.error("❌ Test failed - no subjects were loaded")
            
    except Exception as e:
        logging.error(f"❌ Error in load_timeseries_data: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main() 