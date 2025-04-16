#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(src_path)

from models import initialize_model, AttentionConnectoHopfield

def test_regression_model():
    """Test that the regression model works correctly"""
    # Create a simple configuration
    config = {
        'model': {
            'embedding_dim': 64,
            'hidden_dim': 128,
            'num_encoder_layers': 2,
            'num_attention_heads': 2,
            'dropout': 0.1,
            'embedding_dropout': 0.1,
            'encoder_dropout': 0.1,
            'classifier_dropout': 0.1,
            'use_positional_encoding': True,
            'hopfield_beta': 1.0,
            'hopfield_alpha': 0.5
        },
        'data': {
            'num_regions': 116,
            'sequence_length': 30
        }
    }
    
    logging.info("Initializing regression model...")
    model = initialize_model(config, task_type='regression')
    
    # Check model parameters
    logging.info(f"Model type: {type(model)}")
    logging.info(f"Is regression model: {model.regression}")
    logging.info(f"Output size: {model.output_size}")
    
    # Create a random input tensor
    batch_size = 4
    seq_len = 30
    n_regions = 116
    
    x = torch.randn(batch_size, seq_len, n_regions)
    logging.info(f"Input shape: {x.shape}")
    
    # Forward pass
    logging.info("Running forward pass...")
    output = model(x)
    
    # Check output shape
    logging.info(f"Output shape: {output.shape}")
    logging.info(f"Output: {output}")
    
    # Should be (batch_size,) for regression
    expected_shape = (batch_size,)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Test with return_intermediates=True
    logging.info("Testing with return_intermediates=True...")
    output, attn_weights, intermediates = model(x, return_intermediates=True)
    
    # Check intermediate shapes
    logging.info(f"Output shape: {output.shape}")
    logging.info(f"Attention weights shape: {attn_weights.shape if attn_weights is not None else None}")
    logging.info(f"Intermediate keys: {list(intermediates.keys())}")
    
    logging.info("All tests passed!")

if __name__ == "__main__":
    test_regression_model() 