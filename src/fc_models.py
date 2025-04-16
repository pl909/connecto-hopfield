import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Optional

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for classifying flattened FC vectors.
    """
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_layers: Optional[List[int]] = None,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Initializes the MLP model.

        Args:
            input_dim (int): The number of input features (size of the flattened FC vector).
            num_classes (int): The number of output classes.
            hidden_layers (Optional[List[int]]): A list defining the size of each hidden layer.
                                                  If None or empty, uses a default [128, 64].
            dropout_rate (float): Dropout probability to apply after each hidden layer activation.
            use_batch_norm (bool): Whether to use BatchNorm1d after linear layers (before activation).
        """
        super().__init__()

        if hidden_layers is None or not hidden_layers:
            hidden_layers = [128, 64]
            logging.info(f"No hidden_layers provided, using default: {hidden_layers}")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers_config = hidden_layers
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Log model architecture
        logging.info("=" * 50)
        logging.info("MLP Model Architecture:")
        logging.info("-" * 50)
        logging.info(f"Input Dimension (FC features): {self.input_dim}")
        logging.info(f"Hidden Layers: {self.hidden_layers_config}")
        logging.info(f"Dropout Rate: {self.dropout_rate}")
        logging.info(f"Batch Norm Used: {self.use_batch_norm}")
        logging.info(f"Output Classes: {self.num_classes}")
        logging.info(f"Model Structure:\n{self.network}")
        logging.info("=" * 50)

        # Weight initialization (optional but often helpful)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming initialization for ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Ensure input is float32
        if x.dtype != torch.float32:
           x = x.float()
           
        return self.network(x)
