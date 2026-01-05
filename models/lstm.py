"""
Standard LSTM Baseline

Simple LSTM model for time series prediction.
Serves as the baseline neural network model.

Architecture:
    h_t = LSTM(h_{t-1}, x_t)
    x̂_{t+1} = W_out @ h_t + b_out
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class StandardLSTM(nn.Module):
    """Standard LSTM for time series prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize Standard LSTM.

        Args:
            input_dim: Dimension of input observations
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers if num_layers > 1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM cell(s)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Optional hidden state tuple (h, c) from previous timesteps

        Returns:
            predictions: Predicted next observation (batch_size, input_dim)
            hidden: Final hidden state tuple (h, c)
        """
        # Run LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Predict next observation
        predictions = self.output_layer(last_output)  # (batch_size, input_dim)

        return predictions, hidden

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        """Get detailed parameter count breakdown."""
        lstm_params = sum(p.numel() for name, p in self.named_parameters()
                          if 'lstm' in name and p.requires_grad)
        output_params = sum(p.numel() for name, p in self.named_parameters()
                            if 'output_layer' in name and p.requires_grad)

        return {
            'total': self.count_parameters(),
            'lstm': lstm_params,
            'output_layer': output_params
        }


if __name__ == '__main__':
    # Test LSTM model
    print("Testing Standard LSTM Model")
    print("=" * 60)

    # Create model
    input_dim = 4
    hidden_dim = 64
    batch_size = 32
    seq_len = 10

    model = StandardLSTM(input_dim=input_dim, hidden_dim=hidden_dim)

    print(f"Model created:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {input_dim}")

    # Parameter count
    params = model.get_parameter_breakdown()
    print(f"\nParameter count:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")

    predictions, hidden = model(x)
    print(f"Output shape: {predictions.shape}")
    print(f"Hidden state shape: {hidden[0].shape}, {hidden[1].shape}")

    # Test with different sequence length
    x2 = torch.randn(batch_size, 5, input_dim)
    predictions2, hidden2 = model(x2)
    print(f"\nWith seq_len=5:")
    print(f"  Input shape: {x2.shape}")
    print(f"  Output shape: {predictions2.shape}")

    print("\nTest passed!")
