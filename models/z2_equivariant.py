"""
Z2-Equivariant RNN (No Seam) - Ablation Model

Implements Z2 equivariance through commuting weight matrix,
without the seam gating mechanism.

FIXED INVOLUTION:
    S = [0      I_{n/2}]
        [I_{n/2}  0    ]

PROJECTORS:
    P_+ = (I + S) / 2    [even parity]
    P_- = (I - S) / 2    [odd parity]

EQUIVARIANT UPDATE:
    W_comm = P_+ @ A_+ @ P_+ + P_- @ A_- @ P_-

RECURRENT UPDATE:
    h_{t+1} = tanh(W_comm @ h_t + W_in @ x_t + b)
    x̂_{t+1} = W_out @ h_{t+1} + b_out
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def create_involution(n: int) -> torch.Tensor:
    """
    Create the fixed involution matrix S.

    S = [0      I_{n/2}]
        [I_{n/2}  0    ]

    Args:
        n: Hidden dimension (must be even)

    Returns:
        S: Involution matrix (n, n)
    """
    assert n % 2 == 0, "Hidden dimension must be even for Z2 equivariance"

    n_half = n // 2
    S = torch.zeros(n, n)

    # Upper right: I_{n/2}
    S[:n_half, n_half:] = torch.eye(n_half)
    # Lower left: I_{n/2}
    S[n_half:, :n_half] = torch.eye(n_half)

    return S


def create_projectors(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create parity projectors from involution.

    P_+ = (I + S) / 2    [even parity, +1 eigenspace]
    P_- = (I - S) / 2    [odd parity, -1 eigenspace]

    Args:
        S: Involution matrix

    Returns:
        P_plus: Even parity projector
        P_minus: Odd parity projector
    """
    n = S.shape[0]
    I = torch.eye(n)

    P_plus = (I + S) / 2.0
    P_minus = (I - S) / 2.0

    return P_plus, P_minus


def verify_involution_properties(S: torch.Tensor, atol: float = 1e-6) -> None:
    """
    Verify mathematical properties of involution.

    Args:
        S: Involution matrix
        atol: Absolute tolerance for verification
    """
    n = S.shape[0]
    I = torch.eye(n)

    # S^2 = I
    assert torch.allclose(S @ S, I, atol=atol), "Involution property S^2 = I violated"

    print("✓ Involution property verified: S^2 = I")


def verify_projector_properties(
    P_plus: torch.Tensor,
    P_minus: torch.Tensor,
    atol: float = 1e-6
) -> None:
    """
    Verify mathematical properties of projectors.

    Args:
        P_plus: Even parity projector
        P_minus: Odd parity projector
        atol: Absolute tolerance for verification
    """
    n = P_plus.shape[0]
    I = torch.eye(n)
    Z = torch.zeros(n, n)

    # Idempotence: P^2 = P
    assert torch.allclose(P_plus @ P_plus, P_plus, atol=atol), "P_+ not idempotent"
    assert torch.allclose(P_minus @ P_minus, P_minus, atol=atol), "P_- not idempotent"

    # Orthogonality: P_+ @ P_- = 0
    assert torch.allclose(P_plus @ P_minus, Z, atol=atol), "Projectors not orthogonal"

    # Completeness: P_+ + P_- = I
    assert torch.allclose(P_plus + P_minus, I, atol=atol), "Projectors not complete"

    print("✓ Projector properties verified:")
    print("  - Idempotence: P^2 = P")
    print("  - Orthogonality: P_+ @ P_- = 0")
    print("  - Completeness: P_+ + P_- = I")


def verify_eigenspace_properties(
    S: torch.Tensor,
    P_plus: torch.Tensor,
    P_minus: torch.Tensor,
    atol: float = 1e-6
) -> None:
    """
    Verify eigenspace properties of projectors.

    Args:
        S: Involution matrix
        P_plus: Even parity projector
        P_minus: Odd parity projector
        atol: Absolute tolerance
    """
    # Create test vectors in each eigenspace
    n = S.shape[0]
    h = torch.randn(n)

    h_plus = P_plus @ h
    h_minus = P_minus @ h

    # Even eigenspace: S @ h_+ = h_+
    assert torch.allclose(S @ h_plus, h_plus, atol=atol), "Even eigenspace property violated"

    # Odd eigenspace: S @ h_- = -h_-
    assert torch.allclose(S @ h_minus, -h_minus, atol=atol), "Odd eigenspace property violated"

    print("✓ Eigenspace properties verified:")
    print("  - Even: S @ h_+ = h_+")
    print("  - Odd: S @ h_- = -h_-")


class Z2EquivariantRNN(nn.Module):
    """Z2-Equivariant RNN without seam gating (ablation model)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        verify_properties: bool = True
    ):
        """
        Initialize Z2-Equivariant RNN.

        Args:
            input_dim: Dimension of input observations
            hidden_dim: Dimension of hidden state (must be even)
            verify_properties: Whether to verify mathematical properties
        """
        super().__init__()

        assert hidden_dim % 2 == 0, "Hidden dimension must be even"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Fixed involution and projectors (not learnable)
        self.register_buffer('S', create_involution(hidden_dim))
        P_plus, P_minus = create_projectors(self.S)
        self.register_buffer('P_plus', P_plus)
        self.register_buffer('P_minus', P_minus)

        # Learnable parameters for commuting weight matrix
        self.A_plus = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.A_minus = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # Input and output layers
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, input_dim)

        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        if verify_properties:
            self.verify_mathematical_properties()

    def compute_W_comm(self) -> torch.Tensor:
        """
        Compute the commuting weight matrix.

        W_comm = P_+ @ A_+ @ P_+ + P_- @ A_- @ P_-

        Returns:
            W_comm: Commuting weight matrix
        """
        W_comm = (self.P_plus @ self.A_plus @ self.P_plus +
                  self.P_minus @ self.A_minus @ self.P_minus)
        return W_comm

    def verify_commutativity(self, atol: float = 1e-5) -> None:
        """
        Verify that W_comm commutes with S: [W_comm, S] = 0

        Args:
            atol: Absolute tolerance for verification
        """
        W_comm = self.compute_W_comm()
        commutator = W_comm @ self.S - self.S @ W_comm
        max_error = torch.max(torch.abs(commutator)).item()

        if max_error > atol:
            print(f"⚠ Warning: Commutativity error = {max_error:.2e} > {atol}")
        else:
            print(f"✓ Commutativity verified: max|[W_comm, S]| = {max_error:.2e}")

    def verify_mathematical_properties(self) -> None:
        """Verify all mathematical properties."""
        print("\nVerifying Z2-Equivariant RNN properties:")
        print("=" * 60)

        verify_involution_properties(self.S)
        verify_projector_properties(self.P_plus, self.P_minus)
        verify_eigenspace_properties(self.S, self.P_plus, self.P_minus)
        self.verify_commutativity()

        print("=" * 60)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Z2-equivariant RNN.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Optional hidden state (batch_size, hidden_dim)

        Returns:
            predictions: Predicted next observation (batch_size, input_dim)
            hidden: Final hidden state (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Compute commuting weight matrix (same for all timesteps in batch)
        W_comm = self.compute_W_comm()

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_dim)

            # Recurrent update: h_{t+1} = tanh(W_comm @ h_t + W_in @ x_t + b)
            u = hidden @ W_comm.T + self.W_in(x_t) + self.bias
            hidden = torch.tanh(u)

        # Predict next observation
        predictions = self.W_out(hidden)

        return predictions, hidden

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        """Get detailed parameter count breakdown."""
        return {
            'total': self.count_parameters(),
            'A_plus': self.A_plus.numel(),
            'A_minus': self.A_minus.numel(),
            'W_in': sum(p.numel() for p in self.W_in.parameters()),
            'W_out': sum(p.numel() for p in self.W_out.parameters()),
            'bias': self.bias.numel()
        }


if __name__ == '__main__':
    # Test Z2-Equivariant RNN
    print("Testing Z2-Equivariant RNN (No Seam)")
    print("=" * 60)

    # Create model
    input_dim = 4
    hidden_dim = 64
    batch_size = 32
    seq_len = 10

    model = Z2EquivariantRNN(input_dim=input_dim, hidden_dim=hidden_dim, verify_properties=True)

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
    print(f"Hidden state shape: {hidden.shape}")

    # Verify commutativity after parameter initialization
    model.verify_commutativity()

    # Test parity decomposition
    print("\nTesting parity decomposition:")
    h = hidden[0]  # Take first sample
    h_plus = model.P_plus @ h
    h_minus = model.P_minus @ h
    h_reconstructed = h_plus + h_minus

    print(f"  Original hidden: {h[:5]}")
    print(f"  Reconstructed: {h_reconstructed[:5]}")
    print(f"  Max reconstruction error: {torch.max(torch.abs(h - h_reconstructed)).item():.2e}")

    print("\nTest passed!")
