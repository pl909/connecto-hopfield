#!/usr/bin/env python3
"""
Simple test script to verify that all imports are working correctly.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    """Test imports for regression training"""
    logging.info("Testing imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.append(src_path)
    
    # Try importing from models
    try:
        logging.info("Testing models.py imports...")
        from src.models import initialize_model, AttentionConnectoHopfield
        logging.info("✅ models.py imports succeeded")
    except ImportError as e:
        logging.error(f"❌ models.py imports failed: {e}")
        return
    
    # Try importing from training
    try:
        logging.info("Testing training.py imports...")
        from src.training import (
            train_model_regression, 
            train_epoch_regression,
            validate_epoch_regression,
            evaluate_model_regression,
            configure_optimizer,
            configure_scheduler
        )
        logging.info("✅ training.py imports succeeded")
    except ImportError as e:
        logging.error(f"❌ training.py imports failed: {e}")
        return
    
    # Try importing from data_loader
    try:
        logging.info("Testing data_loader.py imports...")
        from src.data_loader import load_timeseries_data, create_data_loaders
        logging.info("✅ data_loader.py imports succeeded")
    except ImportError as e:
        logging.error(f"❌ data_loader.py imports failed: {e}")
        return
    
    # Try importing from utils
    try:
        logging.info("Testing utils.py imports...")
        from src.utils import set_seed, create_fold_indices, setup_logging
        logging.info("✅ utils.py imports succeeded")
    except ImportError as e:
        logging.error(f"❌ utils.py imports failed: {e}")
        return
    
    logging.info("✅ All imports succeeded! Your code should be ready to run.")

if __name__ == "__main__":
    main() 