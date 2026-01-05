"""
Visualization Code for All Required Figures

Implements:
1. Figure 1: "Money Plot" - Temporal Diagnostics (4-panel)
2. Figure 2: Distribution Comparison (box/violin plots)
3. Figure 3: MSE Decomposition (bar chart with error bars)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_temporal_diagnostics(
    time_range: tuple,
    regimes: np.ndarray,
    parity_energies: np.ndarray,
    gate_activations: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    transition_mask: np.ndarray,
    k_star: float = 0.721347520444,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Figure 1: "Money Plot" - Temporal Diagnostics (4-panel).

    Args:
        time_range: (start, end) indices to plot
        regimes: True regime sequence (T,)
        parity_energies: Parity energy α_-(t) (T,)
        gate_activations: Gate activation g(t) (T,)
        predictions_dict: Dictionary mapping model_name -> predictions (T, obs_dim)
        targets: Ground truth observations (T, obs_dim)
        transition_mask: Binary transition mask (T,)
        k_star: Critical threshold
        save_path: Path to save figure

    Returns:
        Figure object
    """
    start, end = time_range
    t = np.arange(start, end)

    # Find regime switches in this range
    regime_segment = regimes[start:end]
    switches = np.where(np.diff(regime_segment) != 0)[0] + 1
    switch_times = t[switches]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Panel A: True regime
    axes[0].plot(t, regime_segment, 'k-', linewidth=2, label='True regime')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    for switch_t in switch_times:
        axes[0].axvline(switch_t, color='red', linestyle='--', alpha=0.7, linewidth=1)
    axes[0].set_ylabel('Regime $r_t$')
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_yticks([-1, 0, 1])
    axes[0].legend(loc='upper right')
    axes[0].set_title('Panel A: True Regime')

    # Panel B: Parity energy with threshold
    trans_mask_segment = transition_mask[start:end]
    axes[1].plot(t, parity_energies[start:end], 'b-', linewidth=1.5, label='$\\alpha_-(h_t)$')
    axes[1].axhline(k_star, color='red', linestyle='-', linewidth=2, label=f'$k^* = {k_star:.3f}$')

    # Shade transition windows
    axes[1].fill_between(t, 0, 1, where=trans_mask_segment == 1,
                         alpha=0.2, color='orange', label='Transition window',
                         transform=axes[1].get_xaxis_transform())

    axes[1].set_ylabel('Parity Energy')
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Panel B: Parity Energy with Threshold')

    # Panel C: Gate activation
    axes[2].plot(t, gate_activations[start:end], 'g-', linewidth=1.5, label='$g(h_t)$')
    axes[2].axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Threshold = 0.5')

    # Shade transition windows
    axes[2].fill_between(t, 0, 1, where=trans_mask_segment == 1,
                         alpha=0.2, color='orange', label='Transition window',
                         transform=axes[2].get_xaxis_transform())

    axes[2].set_ylabel('Gate Activation')
    axes[2].set_ylim(0, 1)
    axes[2].legend(loc='upper right')
    axes[2].set_title('Panel C: Gate Activation')

    # Panel D: Prediction error
    colors = {'LSTM': 'blue', 'Hamilton/IMM': 'green',
              'Z2-Equiv': 'orange', 'Z2-Seam': 'red'}

    for model_name, preds in predictions_dict.items():
        squared_errors = np.sum((preds[start:end] - targets[start:end]) ** 2, axis=1)
        color = colors.get(model_name, 'black')
        axes[3].plot(t, squared_errors, color=color, linewidth=1.5,
                    label=model_name, alpha=0.8)

    # Shade transition windows
    y_max = axes[3].get_ylim()[1]
    axes[3].fill_between(t, 0, y_max, where=trans_mask_segment == 1,
                         alpha=0.2, color='orange', label='Transition window',
                         transform=axes[3].get_xaxis_transform())

    axes[3].set_ylabel('Squared Error')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='upper right')
    axes[3].set_title('Panel D: Prediction Error')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved temporal diagnostics to {save_path}")

    return fig


def plot_distribution_comparison(
    parity_energies: np.ndarray,
    gate_activations: np.ndarray,
    transition_mask: np.ndarray,
    p_values: Dict[str, float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Figure 2: Distribution Comparison (box/violin plots).

    Args:
        parity_energies: Parity energy time series (T,)
        gate_activations: Gate activation time series (T,)
        transition_mask: Binary transition mask (T,)
        p_values: Dictionary with p-values from statistical tests
        save_path: Path to save figure

    Returns:
        Figure object
    """
    # Prepare data for plotting
    data_alpha = []
    data_gate = []
    labels_alpha = []
    labels_gate = []

    # Parity energy
    data_alpha.extend(parity_energies[transition_mask == 0])
    labels_alpha.extend(['Stable'] * np.sum(transition_mask == 0))
    data_alpha.extend(parity_energies[transition_mask == 1])
    labels_alpha.extend(['Transition'] * np.sum(transition_mask == 1))

    # Gate activation
    data_gate.extend(gate_activations[transition_mask == 0])
    labels_gate.extend(['Stable'] * np.sum(transition_mask == 0))
    data_gate.extend(gate_activations[transition_mask == 1])
    labels_gate.extend(['Transition'] * np.sum(transition_mask == 1))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity energy
    sns.violinplot(x=labels_alpha, y=data_alpha, ax=axes[0], palette='Set2')
    axes[0].set_ylabel('Parity Energy $\\alpha_-(h_t)$')
    axes[0].set_xlabel('')
    title_alpha = 'Parity Energy Distribution'
    if p_values and 'parity_energy' in p_values:
        p_val = p_values['parity_energy']
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        title_alpha += f' (p={p_val:.4f} {sig})'
    axes[0].set_title(title_alpha)

    # Gate activation
    sns.violinplot(x=labels_gate, y=data_gate, ax=axes[1], palette='Set2')
    axes[1].set_ylabel('Gate Activation $g(h_t)$')
    axes[1].set_xlabel('')
    title_gate = 'Gate Activation Distribution'
    if p_values and 'gate_activation' in p_values:
        p_val = p_values['gate_activation']
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        title_gate += f' (p={p_val:.4f} {sig})'
    axes[1].set_title(title_gate)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved distribution comparison to {save_path}")

    return fig


def plot_mse_decomposition(
    results: Dict[str, Dict[str, tuple]],
    model_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Figure 3: MSE Decomposition (bar chart with error bars).

    Args:
        results: Dictionary mapping model_name -> {'mse_overall': (mean, std), ...}
        model_names: Ordered list of model names
        save_path: Path to save figure

    Returns:
        Figure object
    """
    metrics = ['mse_overall', 'mse_stable', 'mse_trans']
    metric_labels = ['Overall', 'Stable', 'Transition']

    x = np.arange(len(metric_labels))
    width = 0.2
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, model_name in enumerate(model_names):
        if model_name not in results:
            continue

        means = []
        stds = []

        for metric in metrics:
            if metric in results[model_name]:
                mean, std = results[model_name][metric]
                means.append(mean if not np.isnan(mean) else 0)
                stds.append(std if not np.isnan(std) else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model_name,
               color=colors[i % len(colors)], capsize=5, alpha=0.8)

    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Evaluation Regime')
    ax.set_title('MSE Decomposition by Model and Regime')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved MSE decomposition to {save_path}")

    return fig


if __name__ == '__main__':
    # Test visualization functions
    print("Testing Visualization Functions")
    print("=" * 60)

    # Create synthetic data
    T = 500
    obs_dim = 4

    # Regimes
    regimes = np.ones(T, dtype=int)
    regimes[100:200] = -1
    regimes[350:] = -1

    # Transition mask
    transition_mask = np.zeros(T, dtype=int)
    transition_mask[95:205] = 1
    transition_mask[345:355] = 1

    # Parity energies (spike at transitions)
    parity_energies = np.random.randn(T) * 0.1 + 0.3
    parity_energies[transition_mask == 1] += 0.5

    # Gate activations
    gate_activations = 1.0 / (1.0 + np.exp(-(parity_energies - 0.721) / 0.05))

    # Predictions
    targets = np.random.randn(T, obs_dim)
    predictions_dict = {
        'LSTM': targets + np.random.randn(T, obs_dim) * 0.3,
        'Hamilton/IMM': targets + np.random.randn(T, obs_dim) * 0.25,
        'Z2-Equiv': targets + np.random.randn(T, obs_dim) * 0.22,
        'Z2-Seam': targets + np.random.randn(T, obs_dim) * 0.18,
    }

    # Test Figure 1
    print("\nTesting Figure 1: Temporal Diagnostics")
    fig1 = plot_temporal_diagnostics(
        time_range=(50, 250),
        regimes=regimes,
        parity_energies=parity_energies,
        gate_activations=gate_activations,
        predictions_dict=predictions_dict,
        targets=targets,
        transition_mask=transition_mask,
        save_path='test_temporal.png'
    )
    plt.close(fig1)

    # Test Figure 2
    print("\nTesting Figure 2: Distribution Comparison")
    p_values = {'parity_energy': 0.001, 'gate_activation': 0.002}
    fig2 = plot_distribution_comparison(
        parity_energies,
        gate_activations,
        transition_mask,
        p_values=p_values,
        save_path='test_distributions.png'
    )
    plt.close(fig2)

    # Test Figure 3
    print("\nTesting Figure 3: MSE Decomposition")
    results = {
        'LSTM': {
            'mse_overall': (0.55, 0.05),
            'mse_stable': (0.45, 0.04),
            'mse_trans': (0.85, 0.08)
        },
        'Hamilton/IMM': {
            'mse_overall': (0.48, 0.04),
            'mse_stable': (0.40, 0.03),
            'mse_trans': (0.75, 0.07)
        },
        'Z2-Equiv': {
            'mse_overall': (0.50, 0.03),
            'mse_stable': (0.42, 0.03),
            'mse_trans': (0.70, 0.06)
        },
        'Z2-Seam': {
            'mse_overall': (0.42, 0.03),
            'mse_stable': (0.38, 0.02),
            'mse_trans': (0.55, 0.05)
        }
    }

    fig3 = plot_mse_decomposition(
        results,
        model_names=['LSTM', 'Hamilton/IMM', 'Z2-Equiv', 'Z2-Seam'],
        save_path='test_mse_decomposition.png'
    )
    plt.close(fig3)

    print("\nTest passed! Check test_*.png files")
