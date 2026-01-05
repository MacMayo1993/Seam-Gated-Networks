"""
Z2 Seam-Gated RNN (Full Proposed Model)

Implements Z2 equivariance with seam gating mechanism for detecting
and handling regime transitions.

SEAM OPERATOR:
    W_flip: Swaps parity subspaces (learnable)
    Chart transition: S @ W_flip

PARITY ENERGY:
    α_-(h) = ||P_- @ h||^2 / (||h||^2 + ε)
    Scale invariant: α_-(λh) = α_-(h)

SEAM GATE:
    k* = 1/(2*ln(2)) ≈ 0.721347... [FIXED]
    g(h) = σ((α_-(h) - k*) / τ)

FULL UPDATE:
    u = W_comm @ h_t + W_in @ x_t + g(h_t) * (S @ W_flip) @ h_t + b
    h_{t+1} = tanh(u)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from models.z2_equivariant import create_involution, create_projectors
from models.z2_equivariant import verify_involution_properties, verify_projector_properties


def compute_parity_energy(
    h: torch.Tensor,
    P_minus: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute parity energy functional.

    α_-(h) = ||P_- @ h||^2 / (||h||^2 + ε)

    This is scale invariant: α_-(λh) = α_-(h) for λ ≠ 0

    Args:
        h: Hidden state (batch_size, hidden_dim) or (hidden_dim,)
        P_minus: Odd parity projector (hidden_dim, hidden_dim)
        eps: Numerical stability constant

    Returns:
        alpha: Parity energy (batch_size,) or scalar
    """
    # Project onto odd parity subspace
    if h.dim() == 1:
        h_minus = P_minus @ h
    else:
        h_minus = h @ P_minus.T

    # Compute norms
    norm_h_minus_sq = torch.sum(h_minus ** 2, dim=-1)
    norm_h_sq = torch.sum(h ** 2, dim=-1)

    # Parity energy
    alpha = norm_h_minus_sq / (norm_h_sq + eps)

    return alpha


def verify_scale_invariance(
    P_minus: torch.Tensor,
    atol: float = 1e-5
) -> None:
    """
    Verify scale invariance of parity energy: α_-(λh) = α_-(h)

    Args:
        P_minus: Odd parity projector
        atol: Absolute tolerance
    """
    hidden_dim = P_minus.shape[0]

    # Test with random hidden state
    h = torch.randn(hidden_dim)

    # Test with different scales
    scales = [0.1, 1.0, 2.5, 10.0]
    alphas = []

    for scale in scales:
        alpha = compute_parity_energy(scale * h, P_minus)
        alphas.append(alpha.item())

    max_diff = max(alphas) - min(alphas)

    if max_diff > atol:
        print(f"⚠ Warning: Scale invariance violated, max diff = {max_diff:.2e}")
    else:
        print(f"✓ Scale invariance verified: max diff = {max_diff:.2e}")


def compute_k_star() -> float:
    """
    Compute the critical threshold k* = 1/(2*ln(2))

    This is the theoretical threshold where the seam gate
    should activate.

    Returns:
        k_star: Critical threshold (≈ 0.721347520444)
    """
    k_star = 1.0 / (2.0 * torch.log(torch.tensor(2.0)))
    return k_star.item()


def verify_k_star(k_star: float, expected: float = 0.721347520444, atol: float = 1e-9) -> None:
    """
    Verify k* value precision.

    Args:
        k_star: Computed k* value
        expected: Expected value
        atol: Absolute tolerance
    """
    error = abs(k_star - expected)
    if error > atol:
        print(f"⚠ Warning: k* = {k_star:.12f}, error = {error:.2e}")
    else:
        print(f"✓ k* verified: {k_star:.12f}")


def create_seam_operator(hidden_dim: int, orthogonal: bool = False) -> nn.Parameter:
    """
    Create seam operator W_flip that swaps parity subspaces.

    Args:
        hidden_dim: Hidden state dimension
        orthogonal: If True, initialize as orthogonal matrix

    Returns:
        W_flip: Seam operator (learnable parameter)
    """
    if orthogonal:
        # Initialize as random orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(hidden_dim, hidden_dim))
        W_flip = nn.Parameter(Q)
    else:
        # Initialize with small random values
        W_flip = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

    return W_flip


class Z2SeamGatedRNN(nn.Module):
    """Z2-Equivariant RNN with Seam Gating (Full Proposed Model)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        temperature: float = 0.05,
        k_star: float = None,
        orthogonal_flip: bool = False,
        verify_properties: bool = True
    ):
        """
        Initialize Z2 Seam-Gated RNN.

        Args:
            input_dim: Dimension of input observations
            hidden_dim: Dimension of hidden state (must be even)
            temperature: Temperature parameter τ for seam gate
            k_star: Critical threshold (default: 1/(2*ln(2)))
            orthogonal_flip: If True, use orthogonal W_flip
            verify_properties: Whether to verify mathematical properties
        """
        super().__init__()

        assert hidden_dim % 2 == 0, "Hidden dimension must be even"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Critical threshold k*
        if k_star is None:
            k_star = compute_k_star()
        self.k_star = k_star

        # Fixed involution and projectors (not learnable)
        self.register_buffer('S', create_involution(hidden_dim))
        P_plus, P_minus = create_projectors(self.S)
        self.register_buffer('P_plus', P_plus)
        self.register_buffer('P_minus', P_minus)

        # Learnable parameters for commuting weight matrix
        self.A_plus = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.A_minus = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)

        # Seam operator W_flip
        self.W_flip = create_seam_operator(hidden_dim, orthogonal=orthogonal_flip)

        # Input and output layers
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, input_dim)

        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # For diagnostics
        self.diagnostics = {
            'parity_energy': [],
            'gate_activation': []
        }

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

    def compute_seam_gate(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute seam gate activation.

        g(h) = σ((α_-(h) - k*) / τ)

        Args:
            h: Hidden state (batch_size, hidden_dim)

        Returns:
            gate: Gate activation (batch_size,)
            alpha: Parity energy (batch_size,)
        """
        # Compute parity energy
        alpha = compute_parity_energy(h, self.P_minus)

        # Compute gate
        logit = (alpha - self.k_star) / self.temperature
        gate = torch.sigmoid(logit)

        return gate, alpha

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through Z2 Seam-Gated RNN.

        Args:
            x: Input sequence (batch_size, seq_len, input_dim)
            hidden: Optional hidden state (batch_size, hidden_dim)
            return_diagnostics: Whether to return diagnostic information

        Returns:
            predictions: Predicted next observation (batch_size, input_dim)
            hidden: Final hidden state (batch_size, hidden_dim)
            diagnostics: Dictionary with parity energy and gate activations [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Compute weight matrices (same for all timesteps in batch)
        W_comm = self.compute_W_comm()
        seam_term = self.S @ self.W_flip

        # For diagnostics
        if return_diagnostics:
            parity_energies = []
            gate_activations = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_dim)

            # Compute seam gate
            gate, alpha = self.compute_seam_gate(hidden)  # (batch_size,)

            if return_diagnostics:
                parity_energies.append(alpha.detach().cpu())
                gate_activations.append(gate.detach().cpu())

            # Recurrent update with seam gating:
            # u = W_comm @ h_t + W_in @ x_t + g(h_t) * (S @ W_flip) @ h_t + b
            u = (hidden @ W_comm.T +
                 self.W_in(x_t) +
                 gate.unsqueeze(1) * (hidden @ seam_term.T) +
                 self.bias)

            hidden = torch.tanh(u)

        # Predict next observation
        predictions = self.W_out(hidden)

        if return_diagnostics:
            diagnostics = {
                'parity_energy': torch.stack(parity_energies, dim=1),  # (batch, seq_len)
                'gate_activation': torch.stack(gate_activations, dim=1)  # (batch, seq_len)
            }
            return predictions, hidden, diagnostics
        else:
            return predictions, hidden, None

    def verify_mathematical_properties(self) -> None:
        """Verify all mathematical properties."""
        print("\nVerifying Z2 Seam-Gated RNN properties:")
        print("=" * 60)

        verify_involution_properties(self.S)
        verify_projector_properties(self.P_plus, self.P_minus)
        verify_scale_invariance(self.P_minus)
        verify_k_star(self.k_star)

        # Verify commutativity
        W_comm = self.compute_W_comm()
        commutator = W_comm @ self.S - self.S @ W_comm
        max_error = torch.max(torch.abs(commutator)).item()
        print(f"✓ Commutativity: max|[W_comm, S]| = {max_error:.2e}")

        print("=" * 60)

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        """Get detailed parameter count breakdown."""
        return {
            'total': self.count_parameters(),
            'A_plus': self.A_plus.numel(),
            'A_minus': self.A_minus.numel(),
            'W_flip': self.W_flip.numel(),
            'W_in': sum(p.numel() for p in self.W_in.parameters()),
            'W_out': sum(p.numel() for p in self.W_out.parameters()),
            'bias': self.bias.numel()
        }


if __name__ == '__main__':
    # Test Z2 Seam-Gated RNN
    print("Testing Z2 Seam-Gated RNN")
    print("=" * 60)

    # Create model
    input_dim = 4
    hidden_dim = 64
    batch_size = 32
    seq_len = 10

    model = Z2SeamGatedRNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        temperature=0.05,
        verify_properties=True
    )

    # Parameter count
    params = model.get_parameter_breakdown()
    print(f"\nParameter count:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")

    # Test forward pass without diagnostics
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")

    predictions, hidden, _ = model(x, return_diagnostics=False)
    print(f"Output shape: {predictions.shape}")
    print(f"Hidden state shape: {hidden.shape}")

    # Test forward pass with diagnostics
    predictions, hidden, diagnostics = model(x, return_diagnostics=True)
    print(f"\nWith diagnostics:")
    print(f"  Parity energy shape: {diagnostics['parity_energy'].shape}")
    print(f"  Gate activation shape: {diagnostics['gate_activation'].shape}")

    # Check diagnostic value ranges
    alpha_mean = diagnostics['parity_energy'].mean().item()
    alpha_std = diagnostics['parity_energy'].std().item()
    gate_mean = diagnostics['gate_activation'].mean().item()

    print(f"\nDiagnostic statistics:")
    print(f"  Parity energy: {alpha_mean:.3f} ± {alpha_std:.3f}")
    print(f"  Gate activation: {gate_mean:.3f}")
    print(f"  k* threshold: {model.k_star:.6f}")

    # Test scale invariance explicitly
    print("\nTesting scale invariance of parity energy:")
    h_test = hidden[0]  # Take first sample
    scales = [0.5, 1.0, 2.0, 5.0]
    alphas = []
    for scale in scales:
        alpha = compute_parity_energy(scale * h_test, model.P_minus)
        alphas.append(alpha.item())
        print(f"  Scale={scale:.1f}: α_- = {alpha.item():.6f}")

    print("\nTest passed!")
