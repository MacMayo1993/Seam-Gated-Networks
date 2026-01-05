"""
Gate Activation Visualization Tools

Provides interpretability by visualizing:
1. Gate activation heatmaps over time
2. Parity energy trajectories
3. Attention-style visualizations of gating patterns
4. Interactive plots for exploration
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, Dict, List
import warnings


class GateActivationHook:
    """
    Hook to extract gate activations during forward pass.

    Usage:
        hook = GateActivationHook()
        model.register_forward_hook(hook)
        output = model(x)
        gates = hook.get_gates()
    """

    def __init__(self, layer_names: Optional[List[str]] = None):
        """
        Initialize hook.

        Args:
            layer_names: Names of layers to monitor (if None, monitor all)
        """
        self.layer_names = layer_names
        self.activations = {}
        self.handles = []

    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Store activations
        module_name = module.__class__.__name__

        if self.layer_names is None or module_name in self.layer_names:
            if isinstance(output, tuple):
                # Handle models that return (predictions, hidden, diagnostics)
                if len(output) == 3 and output[2] is not None:
                    diagnostics = output[2]
                    if 'gate_activation' in diagnostics:
                        self.activations['gate'] = diagnostics['gate_activation'].detach().cpu()
                    if 'parity_energy' in diagnostics:
                        self.activations['parity_energy'] = diagnostics['parity_energy'].detach().cpu()

    def register(self, model: nn.Module):
        """Register hook on model."""
        handle = model.register_forward_hook(self)
        self.handles.append(handle)

    def remove(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get stored activations."""
        return self.activations

    def clear(self):
        """Clear stored activations."""
        self.activations = {}


def plot_gate_heatmap(
    gate_activations: np.ndarray,
    regime_sequence: Optional[np.ndarray] = None,
    transition_mask: Optional[np.ndarray] = None,
    time_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 6),
    cmap: str = 'RdYlGn',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot gate activation heatmap.

    Args:
        gate_activations: Gate values (batch, seq_len) or (seq_len,)
        regime_sequence: True regimes (seq_len,)
        transition_mask: Transition window mask (seq_len,)
        time_range: (start, end) to plot
        figsize: Figure size
        cmap: Colormap name
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Handle different shapes
    if gate_activations.ndim == 1:
        gate_activations = gate_activations[np.newaxis, :]
    elif gate_activations.ndim == 2:
        # Average over batch if needed
        if gate_activations.shape[0] > 1:
            gate_activations = gate_activations.mean(axis=0, keepdims=True)

    # Extract time range
    if time_range is not None:
        start, end = time_range
        gate_activations = gate_activations[:, start:end]
        if regime_sequence is not None:
            regime_sequence = regime_sequence[start:end]
        if transition_mask is not None:
            transition_mask = transition_mask[start:end]
    else:
        start = 0
        end = gate_activations.shape[1]

    # Create figure
    if regime_sequence is not None or transition_mask is not None:
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 3, 1], sharex=True)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [None, ax, None]

    # Plot regime sequence
    if regime_sequence is not None and axes[0] is not None:
        t = np.arange(start, end)
        axes[0].plot(t, regime_sequence, 'k-', linewidth=2)
        axes[0].set_ylabel('Regime', fontsize=12)
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].set_yticks([-1, 0, 1])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('True Regime Sequence', fontsize=14, fontweight='bold')

    # Plot gate heatmap
    ax = axes[1]
    im = ax.imshow(
        gate_activations,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[start, end, 0, 1],
        interpolation='nearest'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Gate Activation', fontsize=12)

    # Overlay transition windows if provided
    if transition_mask is not None:
        trans_idx = np.where(transition_mask == 1)[0]
        for idx in trans_idx:
            if start <= idx < end:
                ax.axvline(idx, color='orange', alpha=0.3, linewidth=1)

    ax.set_ylabel('Hidden Dim\n(averaged)', fontsize=12)
    ax.set_title('Gate Activation Heatmap', fontsize=14, fontweight='bold')

    # Plot transition mask
    if transition_mask is not None and axes[2] is not None:
        t = np.arange(start, end)
        axes[2].fill_between(t, 0, 1, where=transition_mask == 1,
                            alpha=0.5, color='orange', label='Transition')
        axes[2].fill_between(t, 0, 1, where=transition_mask == 0,
                            alpha=0.5, color='lightblue', label='Stable')
        axes[2].set_ylabel('Window', fontsize=12)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].set_ylim(0, 1)
        axes[2].set_yticks([])
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].set_title('Transition Windows', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved gate heatmap to {save_path}")

    return fig


def plot_parity_energy_trajectory(
    parity_energies: np.ndarray,
    gate_activations: np.ndarray,
    k_star: float = 0.721347520444,
    regime_sequence: Optional[np.ndarray] = None,
    time_range: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot parity energy and gate activation trajectories.

    Args:
        parity_energies: Parity energy values (batch, seq_len) or (seq_len,)
        gate_activations: Gate values (batch, seq_len) or (seq_len,)
        k_star: Critical threshold
        regime_sequence: True regimes (seq_len,)
        time_range: (start, end) to plot
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Handle shapes
    if parity_energies.ndim == 2:
        parity_energies = parity_energies.mean(axis=0)
    if gate_activations.ndim == 2:
        gate_activations = gate_activations.mean(axis=0)

    # Extract time range
    if time_range is not None:
        start, end = time_range
        parity_energies = parity_energies[start:end]
        gate_activations = gate_activations[start:end]
        if regime_sequence is not None:
            regime_sequence = regime_sequence[start:end]
    else:
        start = 0
        end = len(parity_energies)

    t = np.arange(start, end)

    # Create figure
    n_plots = 3 if regime_sequence is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    # Plot regime (if provided)
    if regime_sequence is not None:
        axes[0].plot(t, regime_sequence, 'k-', linewidth=2)
        axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Regime $r_t$', fontsize=12)
        axes[0].set_ylim(-1.5, 1.5)
        axes[0].set_yticks([-1, 0, 1])
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('A: True Regime Sequence', fontsize=13, fontweight='bold')

        ax_idx = 1
    else:
        ax_idx = 0

    # Plot parity energy
    axes[ax_idx].plot(t, parity_energies, 'b-', linewidth=1.5, label='$\\alpha_-(h_t)$')
    axes[ax_idx].axhline(k_star, color='red', linestyle='-', linewidth=2,
                        label=f'$k^* = {k_star:.3f}$')

    # Shade regions where energy crosses threshold
    above_threshold = parity_energies > k_star
    axes[ax_idx].fill_between(t, 0, 1, where=above_threshold,
                             alpha=0.2, color='red',
                             transform=axes[ax_idx].get_xaxis_transform())

    axes[ax_idx].set_ylabel('Parity Energy', fontsize=12)
    axes[ax_idx].set_ylim(0, 1)
    axes[ax_idx].grid(True, alpha=0.3)
    axes[ax_idx].legend(loc='upper right', fontsize=11)
    axes[ax_idx].set_title('B: Parity Energy with Critical Threshold',
                          fontsize=13, fontweight='bold')

    # Plot gate activation
    axes[ax_idx + 1].plot(t, gate_activations, 'g-', linewidth=1.5, label='$g(h_t)$')
    axes[ax_idx + 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5,
                            label='Activation threshold')

    # Shade activated regions
    activated = gate_activations > 0.5
    axes[ax_idx + 1].fill_between(t, 0, 1, where=activated,
                                  alpha=0.2, color='green',
                                  transform=axes[ax_idx + 1].get_xaxis_transform())

    axes[ax_idx + 1].set_ylabel('Gate Activation', fontsize=12)
    axes[ax_idx + 1].set_ylim(0, 1)
    axes[ax_idx + 1].set_xlabel('Time', fontsize=12)
    axes[ax_idx + 1].grid(True, alpha=0.3)
    axes[ax_idx + 1].legend(loc='upper right', fontsize=11)
    axes[ax_idx + 1].set_title('C: Seam Gate Activation',
                              fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved parity energy trajectory to {save_path}")

    return fig


def plot_gate_statistics(
    gate_activations: np.ndarray,
    transition_mask: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot statistical comparison of gate activations.

    Args:
        gate_activations: Gate values (seq_len,) or (batch, seq_len)
        transition_mask: Binary transition mask (seq_len,)
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Handle shapes
    if gate_activations.ndim == 2:
        gate_activations = gate_activations.mean(axis=0)

    # Split by mask
    stable_gates = gate_activations[transition_mask == 0]
    trans_gates = gate_activations[transition_mask == 1]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Violin plot
    data = [stable_gates, trans_gates]
    positions = [1, 2]
    parts = axes[0].violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    colors = ['lightblue', 'orange']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(['Stable', 'Transition'])
    axes[0].set_ylabel('Gate Activation', fontsize=12)
    axes[0].set_title('Distribution Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add statistics
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(trans_gates, stable_gates, equal_var=False)

    axes[0].text(0.5, 0.95, f'Welch\'s t-test: p={p_val:.6f}',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram
    axes[1].hist(stable_gates, bins=50, alpha=0.6, label='Stable', color='lightblue', density=True)
    axes[1].hist(trans_gates, bins=50, alpha=0.6, label='Transition', color='orange', density=True)
    axes[1].axvline(stable_gates.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Stable mean: {stable_gates.mean():.3f}')
    axes[1].axvline(trans_gates.mean(), color='darkorange', linestyle='--', linewidth=2,
                   label=f'Trans. mean: {trans_gates.mean():.3f}')
    axes[1].set_xlabel('Gate Activation', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_title('Histogram Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved gate statistics to {save_path}")

    return fig


if __name__ == '__main__':
    # Test visualization tools
    print("Testing Gate Visualization Tools")
    print("=" * 80)

    # Create synthetic data
    seq_len = 500
    batch_size = 16

    # Simulate gate activations (higher in middle)
    gate_acts = np.random.randn(batch_size, seq_len) * 0.1 + 0.3
    gate_acts[:, 200:300] += 0.5  # Spike in middle

    # Simulate parity energies
    parity = np.random.randn(batch_size, seq_len) * 0.1 + 0.4
    parity[:, 200:300] += 0.3

    # Regime sequence
    regimes = np.ones(seq_len)
    regimes[250:] = -1

    # Transition mask
    trans_mask = np.zeros(seq_len)
    trans_mask[245:255] = 1

    print("\nPlotting gate heatmap...")
    fig1 = plot_gate_heatmap(
        gate_acts,
        regime_sequence=regimes,
        transition_mask=trans_mask,
        time_range=(100, 400),
        save_path='test_gate_heatmap.png'
    )
    plt.close(fig1)

    print("\nPlotting parity energy trajectory...")
    fig2 = plot_parity_energy_trajectory(
        parity,
        gate_acts,
        regime_sequence=regimes,
        time_range=(100, 400),
        save_path='test_parity_trajectory.png'
    )
    plt.close(fig2)

    print("\nPlotting gate statistics...")
    fig3 = plot_gate_statistics(
        gate_acts.mean(axis=0),
        trans_mask,
        save_path='test_gate_statistics.png'
    )
    plt.close(fig3)

    print("\n✓ All visualization tests passed!")
    print("Check test_*.png files")
