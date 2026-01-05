"""
Evaluation Harness with MSE Metrics Decomposition

Computes evaluation metrics:
- MSE_overall: Overall mean squared error
- MSE_stable: MSE on stable (non-transition) windows
- MSE_trans: MSE on transition windows

All models use the SAME transition mask for fair comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader
from experiments.train import TimeSeriesDataset


def compute_mse_decomposition(
    predictions: np.ndarray,
    targets: np.ndarray,
    transition_mask: np.ndarray,
    indices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute MSE decomposed by transition/stable windows.

    Args:
        predictions: Model predictions (N, obs_dim)
        targets: Ground truth targets (N, obs_dim)
        transition_mask: Binary mask for entire series (T,)
        indices: Indices corresponding to predictions (N,)
                If None, assumes predictions correspond to indices [0, ..., N-1]

    Returns:
        Dictionary with MSE_overall, MSE_stable, MSE_trans, n_stable, n_trans
    """
    N = len(predictions)

    if indices is None:
        indices = np.arange(N)

    # Compute squared errors
    squared_errors = np.sum((predictions - targets) ** 2, axis=1)  # (N,)

    # Get transition mask for these indices
    mask_subset = transition_mask[indices]

    # Split into transition and stable
    transition_idx = mask_subset == 1
    stable_idx = mask_subset == 0

    # Compute MSEs
    mse_overall = np.mean(squared_errors)

    if np.any(stable_idx):
        mse_stable = np.mean(squared_errors[stable_idx])
        n_stable = np.sum(stable_idx)
    else:
        mse_stable = np.nan
        n_stable = 0

    if np.any(transition_idx):
        mse_trans = np.mean(squared_errors[transition_idx])
        n_trans = np.sum(transition_idx)
    else:
        mse_trans = np.nan
        n_trans = 0

    return {
        'mse_overall': mse_overall,
        'mse_stable': mse_stable,
        'mse_trans': mse_trans,
        'n_stable': n_stable,
        'n_trans': n_trans,
        'n_total': N
    }


def evaluate_neural_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    transition_mask: np.ndarray,
    test_indices: np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate a trained neural network model.

    Args:
        model: Trained model
        X_test: Test input sequences (N, history_length, input_dim)
        y_test: Test targets (N, input_dim)
        transition_mask: Binary transition mask for entire series
        test_indices: Indices corresponding to test samples
        batch_size: Batch size for evaluation
        device: Device to evaluate on
        return_predictions: Whether to return predictions

    Returns:
        Dictionary with evaluation metrics (and predictions if requested)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # Create dataloader
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Generate predictions
    all_predictions = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)

            # Forward pass
            output = model(X_batch)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output

            all_predictions.append(predictions.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)

    # Compute MSE decomposition
    metrics = compute_mse_decomposition(
        predictions,
        y_test,
        transition_mask,
        test_indices
    )

    if return_predictions:
        metrics['predictions'] = predictions

    return metrics


def evaluate_hamilton_imm(
    filter_obj,
    observations: np.ndarray,
    transition_mask: np.ndarray,
    test_indices: np.ndarray,
    return_predictions: bool = False,
    return_diagnostics: bool = False
) -> Dict:
    """
    Evaluate Hamilton/IMM filter.

    Args:
        filter_obj: Hamilton/IMM filter instance
        observations: Full observation sequence (T, obs_dim)
        transition_mask: Binary transition mask for entire series
        test_indices: Indices to evaluate on
        return_predictions: Whether to return predictions
        return_diagnostics: Whether to return regime diagnostics

    Returns:
        Dictionary with evaluation metrics
    """
    # Reset filter and run on full sequence
    filter_obj.reset()
    predictions, diagnostics = filter_obj.filter_sequence(
        observations,
        return_diagnostics=True
    )

    # Extract predictions for test indices
    # Note: predictions[i] corresponds to observation i+1
    # So predictions[t-1] predicts observations[t]
    # For test_indices, we need predictions at indices test_indices - 1

    # Shift test indices back by 1 to get prediction indices
    pred_indices = test_indices - 1

    # Filter out any negative indices (shouldn't happen if test_indices >= 1)
    valid_mask = pred_indices >= 0
    pred_indices = pred_indices[valid_mask]

    # Get predictions and targets
    test_predictions = predictions[pred_indices]
    test_targets = observations[test_indices[valid_mask]]

    # Compute MSE decomposition
    metrics = compute_mse_decomposition(
        test_predictions,
        test_targets,
        transition_mask,
        test_indices[valid_mask]
    )

    # Add regime classification metrics if true regimes available
    if return_diagnostics:
        metrics['regime_probs'] = diagnostics['regime_probs']
        metrics['map_regimes'] = diagnostics['map_regimes']

    if return_predictions:
        metrics['predictions'] = test_predictions

    return metrics


def aggregate_metrics_over_seeds(
    metrics_list: list
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics over multiple seeds.

    Args:
        metrics_list: List of metric dictionaries from different seeds

    Returns:
        Dictionary mapping metric name -> (mean, std)
    """
    # Extract metric names (excluding non-numeric fields)
    numeric_keys = ['mse_overall', 'mse_stable', 'mse_trans']

    aggregated = {}

    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]

        if len(values) > 0:
            aggregated[key] = (np.mean(values), np.std(values))
        else:
            aggregated[key] = (np.nan, np.nan)

    return aggregated


def format_metrics_table(
    results: Dict[str, Dict],
    model_names: list = None
) -> str:
    """
    Format evaluation results as a table.

    Args:
        results: Dictionary mapping model_name -> aggregated metrics
        model_names: Ordered list of model names (default: sorted keys)

    Returns:
        Formatted table string
    """
    if model_names is None:
        model_names = sorted(results.keys())

    # Header
    table = "=" * 80 + "\n"
    table += f"{'Model':<25} | {'MSE (Overall)':<18} | {'MSE (Stable)':<18} | {'MSE (Trans.)':<18}\n"
    table += "=" * 80 + "\n"

    # Rows
    for model_name in model_names:
        if model_name not in results:
            continue

        metrics = results[model_name]

        # Format each metric as mean ± std
        def format_metric(key):
            if key in metrics:
                mean, std = metrics[key]
                if np.isnan(mean):
                    return "N/A"
                return f"{mean:.6f}±{std:.6f}"
            return "N/A"

        mse_overall = format_metric('mse_overall')
        mse_stable = format_metric('mse_stable')
        mse_trans = format_metric('mse_trans')

        table += f"{model_name:<25} | {mse_overall:<18} | {mse_stable:<18} | {mse_trans:<18}\n"

    table += "=" * 80 + "\n"

    return table


if __name__ == '__main__':
    # Test evaluation functions
    print("Testing Evaluation Harness")
    print("=" * 60)

    # Create dummy data
    N = 500
    obs_dim = 4

    predictions = np.random.randn(N, obs_dim)
    targets = np.random.randn(N, obs_dim)

    # Create dummy transition mask
    T = 1000
    transition_mask = np.zeros(T, dtype=int)
    transition_mask[100:150] = 1  # Transition window
    transition_mask[400:450] = 1  # Another transition window

    test_indices = np.arange(500, 1000)  # Test on second half

    print(f"Predictions shape: {predictions.shape}")
    print(f"Transition mask shape: {transition_mask.shape}")
    print(f"Test indices: {len(test_indices)}")

    # Compute MSE decomposition
    metrics = compute_mse_decomposition(
        predictions,
        targets,
        transition_mask,
        test_indices
    )

    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test aggregation
    print("\nTesting metric aggregation:")
    metrics_list = [
        {'mse_overall': 0.5, 'mse_stable': 0.4, 'mse_trans': 0.8},
        {'mse_overall': 0.6, 'mse_stable': 0.5, 'mse_trans': 0.9},
        {'mse_overall': 0.55, 'mse_stable': 0.45, 'mse_trans': 0.85},
    ]

    aggregated = aggregate_metrics_over_seeds(metrics_list)
    print("Aggregated metrics:")
    for key, (mean, std) in aggregated.items():
        print(f"  {key}: {mean:.4f} ± {std:.4f}")

    # Test table formatting
    print("\nTesting table formatting:")
    results = {
        'LSTM': {
            'mse_overall': (0.55, 0.05),
            'mse_stable': (0.45, 0.04),
            'mse_trans': (0.85, 0.08)
        },
        'Z2-Seam': {
            'mse_overall': (0.45, 0.03),
            'mse_stable': (0.40, 0.02),
            'mse_trans': (0.65, 0.05)
        }
    }

    table = format_metrics_table(results, model_names=['LSTM', 'Z2-Seam'])
    print(table)

    print("Test passed!")
