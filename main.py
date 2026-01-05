"""
Main Experimental Pipeline

Runs complete experimental pipeline:
1. Generate data with antipodal regime switching
2. Train all models (LSTM, Hamilton/IMM, Z2-Equiv, Z2-Seam)
3. Evaluate with MSE decomposition
4. Run statistical tests for falsifiable predictions
5. Generate visualizations
6. Save results table
"""

import argparse
import json
import os
import numpy as np
import torch
from typing import Dict
import sys

# Import data generation
from data.generator import AntipodalRegimeSwitchingGenerator, create_supervised_dataset
from data.transition_mask import compute_transition_mask, compute_transition_statistics

# Import models
from models.lstm import StandardLSTM
from models.hamilton_imm import HamiltonIMM
from models.z2_equivariant import Z2EquivariantRNN
from models.z2_seam_gated import Z2SeamGatedRNN

# Import experiment modules
from experiments.train import train_model
from experiments.evaluate import (
    evaluate_neural_model,
    evaluate_hamilton_imm,
    aggregate_metrics_over_seeds,
    format_metrics_table
)
from experiments.diagnostics import (
    prediction1_parity_energy_peaks,
    prediction2_selective_gate_activation,
    prediction3_boundary_specific_performance,
    print_diagnostic_results
)
from experiments.visualize import (
    plot_temporal_diagnostics,
    plot_distribution_comparison,
    plot_mse_decomposition
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_data(config: dict) -> Dict:
    """
    Generate synthetic data with antipodal regime switching.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with data and metadata
    """
    print("\n" + "=" * 80)
    print("GENERATING DATA")
    print("=" * 80)

    # Create generator
    generator = AntipodalRegimeSwitchingGenerator(
        latent_dim=config['latent_dim'],
        obs_dim=config['obs_dim'],
        p_switch=config['p_switch'],
        sigma_epsilon=config['sigma_epsilon'],
        sigma_eta=config['sigma_eta'],
        seed=config['seed']
    )

    # Generate time series
    data = generator.generate(T=config['T'], return_latent=True)

    print(f"Generated time series with T={config['T']} timesteps")
    print(f"Observation dimension: {config['obs_dim']}")
    print(f"Latent dimension: {config['latent_dim']}")

    # Compute transition mask
    transition_mask, switch_indices = compute_transition_mask(
        data['regimes'],
        window_radius=config['window_radius']
    )

    stats = compute_transition_statistics(data['regimes'], transition_mask)
    print(f"\nTransition statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Create supervised dataset
    X, y = create_supervised_dataset(data['x'], history_length=config['history_length'])

    print(f"\nSupervised dataset:")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    # Split into train/val/test
    n_total = len(X)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    # Test indices correspond to prediction targets
    # X[i] predicts y[i] which is x[i + history_length]
    test_indices = np.arange(n_train + n_val + config['history_length'],
                             config['T'])

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    return {
        'generator': generator,
        'data': data,
        'transition_mask': transition_mask,
        'switch_indices': switch_indices,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'test_indices': test_indices
    }


def train_all_models(dataset: Dict, config: dict, device: torch.device) -> Dict:
    """
    Train all neural network models.

    Args:
        dataset: Dataset dictionary
        config: Configuration dictionary
        device: Device to train on

    Returns:
        Dictionary of trained models
    """
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    models = {}

    # 1. Standard LSTM
    print("\n1. Training Standard LSTM")
    print("-" * 80)
    lstm = StandardLSTM(
        input_dim=config['obs_dim'],
        hidden_dim=config['hidden_dim']
    )
    print(f"Parameters: {lstm.count_parameters():,}")

    history_lstm = train_model(
        lstm,
        dataset['X_train'], dataset['y_train'],
        dataset['X_val'], dataset['y_val'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device,
        verbose=True
    )
    models['LSTM'] = lstm
    print(f"Best validation loss: {history_lstm['best_val_loss']:.6f}")

    # 2. Z2-Equivariant (no seam)
    print("\n2. Training Z2-Equivariant RNN (No Seam)")
    print("-" * 80)
    z2_equiv = Z2EquivariantRNN(
        input_dim=config['obs_dim'],
        hidden_dim=config['hidden_dim'],
        verify_properties=False  # Already verified in __init__
    )
    print(f"Parameters: {z2_equiv.count_parameters():,}")

    history_z2 = train_model(
        z2_equiv,
        dataset['X_train'], dataset['y_train'],
        dataset['X_val'], dataset['y_val'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device,
        verbose=True
    )
    models['Z2-Equiv'] = z2_equiv
    print(f"Best validation loss: {history_z2['best_val_loss']:.6f}")

    # 3. Z2 Seam-Gated
    print("\n3. Training Z2 Seam-Gated RNN")
    print("-" * 80)
    z2_seam = Z2SeamGatedRNN(
        input_dim=config['obs_dim'],
        hidden_dim=config['hidden_dim'],
        temperature=config['temperature'],
        verify_properties=False  # Already verified in __init__
    )
    print(f"Parameters: {z2_seam.count_parameters():,}")

    history_seam = train_model(
        z2_seam,
        dataset['X_train'], dataset['y_train'],
        dataset['X_val'], dataset['y_val'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device,
        verbose=True
    )
    models['Z2-Seam'] = z2_seam
    print(f"Best validation loss: {history_seam['best_val_loss']:.6f}")

    return models


def create_hamilton_imm_filter(dataset: Dict, config: dict) -> HamiltonIMM:
    """
    Create Hamilton/IMM filter with known parameters.

    Args:
        dataset: Dataset dictionary
        config: Configuration dictionary

    Returns:
        Hamilton/IMM filter instance
    """
    print("\n" + "=" * 80)
    print("CREATING HAMILTON/IMM FILTER")
    print("=" * 80)

    # Get system matrices from generator
    matrices = dataset['generator'].get_system_matrices()

    # Create filter
    filter_obj = HamiltonIMM(
        A=matrices['A'],
        C=matrices['C'],
        Q=matrices['Q'],
        R=matrices['R'],
        p_switch=config['p_switch']
    )

    print("Hamilton/IMM filter created with known parameters")
    print(f"  Latent dim: {matrices['A'].shape[0]}")
    print(f"  Obs dim: {matrices['C'].shape[0]}")
    print(f"  p_switch: {config['p_switch']}")

    return filter_obj


def evaluate_all_models(
    models: Dict,
    hamilton_filter: HamiltonIMM,
    dataset: Dict,
    config: dict,
    device: torch.device
) -> Dict:
    """
    Evaluate all models on test set.

    Args:
        models: Dictionary of trained models
        hamilton_filter: Hamilton/IMM filter
        dataset: Dataset dictionary
        config: Configuration dictionary
        device: Device for evaluation

    Returns:
        Dictionary of evaluation results
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)

    results = {}

    # Evaluate neural models
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_neural_model(
            model,
            dataset['X_test'],
            dataset['y_test'],
            dataset['transition_mask'],
            dataset['test_indices'],
            device=device,
            return_predictions=True
        )
        results[name] = metrics

        print(f"  MSE overall: {metrics['mse_overall']:.6f}")
        print(f"  MSE stable: {metrics['mse_stable']:.6f}")
        print(f"  MSE transition: {metrics['mse_trans']:.6f}")

    # Evaluate Hamilton/IMM
    print(f"\nEvaluating Hamilton/IMM...")
    metrics_hamilton = evaluate_hamilton_imm(
        hamilton_filter,
        dataset['data']['x'],
        dataset['transition_mask'],
        dataset['test_indices'],
        return_predictions=True,
        return_diagnostics=True
    )
    results['Hamilton/IMM'] = metrics_hamilton

    # Compute regime classification accuracy
    true_regimes = dataset['data']['regimes']
    accuracy = hamilton_filter.get_regime_classification_accuracy(
        metrics_hamilton['map_regimes'],
        true_regimes
    )

    print(f"  MSE overall: {metrics_hamilton['mse_overall']:.6f}")
    print(f"  MSE stable: {metrics_hamilton['mse_stable']:.6f}")
    print(f"  MSE transition: {metrics_hamilton['mse_trans']:.6f}")
    print(f"  Regime accuracy: {accuracy:.3f}")

    return results


def run_diagnostics(
    models: Dict,
    results: Dict,
    dataset: Dict,
    config: dict,
    device: torch.device
) -> Dict:
    """
    Run statistical tests for falsifiable predictions.

    Args:
        models: Dictionary of trained models
        results: Evaluation results
        dataset: Dataset dictionary
        config: Configuration dictionary
        device: Device

    Returns:
        Dictionary of diagnostic test results
    """
    print("\n" + "=" * 80)
    print("RUNNING DIAGNOSTIC TESTS")
    print("=" * 80)

    # Extract diagnostics from Z2-Seam model
    z2_seam = models['Z2-Seam']
    z2_seam.eval()

    # Run forward pass to get diagnostics
    X_full = torch.FloatTensor(
        create_supervised_dataset(
            dataset['data']['x'],
            history_length=config['history_length']
        )[0]
    ).to(device)

    with torch.no_grad():
        _, _, diagnostics = z2_seam(X_full, return_diagnostics=True)

    # Average over batch dimension to get time series
    parity_energies = diagnostics['parity_energy'].mean(dim=0).cpu().numpy()
    gate_activations = diagnostics['gate_activation'].mean(dim=0).cpu().numpy()

    # Note: diagnostics[i] corresponds to processing X[i] which predicts y[i]
    # y[i] = x[i + history_length]
    # So parity_energies[i] is computed at time i + history_length - 1

    # Adjust indices to align with full time series
    diag_start_idx = config['history_length'] - 1
    diag_indices = np.arange(diag_start_idx, diag_start_idx + len(parity_energies))

    # Create full time series (pad beginning with NaN)
    full_parity = np.full(len(dataset['data']['regimes']), np.nan)
    full_parity[diag_indices] = parity_energies

    full_gate = np.full(len(dataset['data']['regimes']), np.nan)
    full_gate[diag_indices] = gate_activations

    # Test only on valid indices (where we have diagnostics)
    valid_mask = ~np.isnan(full_parity)
    transition_mask_valid = dataset['transition_mask'][valid_mask]
    parity_valid = full_parity[valid_mask]
    gate_valid = full_gate[valid_mask]

    diagnostic_results = {}

    # Prediction 1: Parity energy peaks at transitions
    result1 = prediction1_parity_energy_peaks(parity_valid, transition_mask_valid)
    print_diagnostic_results(result1)
    diagnostic_results['prediction1'] = result1

    # Prediction 2: Selective gate activation
    result2 = prediction2_selective_gate_activation(gate_valid, transition_mask_valid)
    print_diagnostic_results(result2)
    diagnostic_results['prediction2'] = result2

    # Prediction 3: Boundary-specific performance (vs Hamilton/IMM)
    result3 = prediction3_boundary_specific_performance(
        mse_trans_hamilton=results['Hamilton/IMM']['mse_trans'],
        mse_trans_seam=results['Z2-Seam']['mse_trans'],
        mse_stable_hamilton=results['Hamilton/IMM']['mse_stable'],
        mse_stable_seam=results['Z2-Seam']['mse_stable'],
        predictions_hamilton=results['Hamilton/IMM']['predictions'],
        predictions_seam=results['Z2-Seam']['predictions'],
        targets=dataset['y_test'],
        transition_mask=dataset['transition_mask'],
        test_indices=dataset['test_indices'],
        n_bootstrap=1000,
        random_seed=config['seed']
    )
    print_diagnostic_results(result3)
    diagnostic_results['prediction3'] = result3

    # Store diagnostics for visualization
    diagnostic_results['parity_energies'] = full_parity
    diagnostic_results['gate_activations'] = full_gate

    return diagnostic_results


def generate_visualizations(
    models: Dict,
    results: Dict,
    diagnostics: Dict,
    dataset: Dict,
    config: dict,
    save_dir: str
):
    """
    Generate all required figures.

    Args:
        models: Dictionary of trained models
        results: Evaluation results
        diagnostics: Diagnostic test results
        dataset: Dataset dictionary
        config: Configuration dictionary
        save_dir: Directory to save figures
    """
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    os.makedirs(save_dir, exist_ok=True)

    # Figure 1: Temporal diagnostics (show 300 timesteps from test set)
    test_start = dataset['test_indices'][0]
    plot_range = (test_start, min(test_start + 300, len(dataset['data']['regimes'])))

    predictions_dict = {
        'LSTM': results['LSTM']['predictions'],
        'Hamilton/IMM': results['Hamilton/IMM']['predictions'],
        'Z2-Equiv': results['Z2-Equiv']['predictions'],
        'Z2-Seam': results['Z2-Seam']['predictions']
    }

    # Reconstruct full predictions (pad with NaN)
    full_predictions = {}
    for name, preds in predictions_dict.items():
        full_preds = np.full((len(dataset['data']['regimes']), config['obs_dim']), np.nan)
        full_preds[dataset['test_indices']] = preds
        full_predictions[name] = full_preds

    fig1 = plot_temporal_diagnostics(
        time_range=plot_range,
        regimes=dataset['data']['regimes'],
        parity_energies=diagnostics['parity_energies'],
        gate_activations=diagnostics['gate_activations'],
        predictions_dict=full_predictions,
        targets=dataset['data']['x'],
        transition_mask=dataset['transition_mask'],
        save_path=os.path.join(save_dir, 'figure1_temporal_diagnostics.png')
    )

    # Figure 2: Distribution comparison
    valid_mask = ~np.isnan(diagnostics['parity_energies'])
    p_values = {
        'parity_energy': diagnostics['prediction1']['p_value'],
        'gate_activation': diagnostics['prediction2']['p_value']
    }

    fig2 = plot_distribution_comparison(
        diagnostics['parity_energies'][valid_mask],
        diagnostics['gate_activations'][valid_mask],
        dataset['transition_mask'][valid_mask],
        p_values=p_values,
        save_path=os.path.join(save_dir, 'figure2_distributions.png')
    )

    # Figure 3: MSE decomposition (convert to aggregated format)
    agg_results = {}
    for name, metrics in results.items():
        agg_results[name] = {
            'mse_overall': (metrics['mse_overall'], 0.0),
            'mse_stable': (metrics['mse_stable'], 0.0),
            'mse_trans': (metrics['mse_trans'], 0.0)
        }

    fig3 = plot_mse_decomposition(
        agg_results,
        model_names=['LSTM', 'Hamilton/IMM', 'Z2-Equiv', 'Z2-Seam'],
        save_path=os.path.join(save_dir, 'figure3_mse_decomposition.png')
    )

    print(f"Figures saved to {save_dir}")


def save_results(
    results: Dict,
    diagnostics: Dict,
    config: dict,
    save_dir: str
):
    """
    Save results to disk.

    Args:
        results: Evaluation results
        diagnostics: Diagnostic test results
        config: Configuration dictionary
        save_dir: Directory to save results
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs(save_dir, exist_ok=True)

    # Save metrics table
    agg_results = {}
    for name, metrics in results.items():
        agg_results[name] = {
            'mse_overall': (metrics['mse_overall'], 0.0),
            'mse_stable': (metrics['mse_stable'], 0.0),
            'mse_trans': (metrics['mse_trans'], 0.0)
        }

    table = format_metrics_table(
        agg_results,
        model_names=['LSTM', 'Hamilton/IMM', 'Z2-Equiv', 'Z2-Seam']
    )

    with open(os.path.join(save_dir, 'results_table.txt'), 'w') as f:
        f.write(table)

    print(table)

    # Save detailed results as JSON
    results_json = {
        'config': config,
        'metrics': {
            name: {
                'mse_overall': float(m['mse_overall']),
                'mse_stable': float(m['mse_stable']),
                'mse_trans': float(m['mse_trans']),
                'n_total': int(m['n_total']),
                'n_stable': int(m['n_stable']),
                'n_trans': int(m['n_trans'])
            }
            for name, m in results.items()
        },
        'diagnostics': {
            'prediction1': {
                'test_name': diagnostics['prediction1']['test_name'],
                'p_value': float(diagnostics['prediction1']['p_value']),
                'significant': bool(diagnostics['prediction1']['significant']),
                'mean_trans': float(diagnostics['prediction1']['mean1']),
                'mean_stable': float(diagnostics['prediction1']['mean2'])
            },
            'prediction2': {
                'test_name': diagnostics['prediction2']['test_name'],
                'p_value': float(diagnostics['prediction2']['p_value']),
                'significant': bool(diagnostics['prediction2']['significant']),
                'mean_trans': float(diagnostics['prediction2']['mean1']),
                'mean_stable': float(diagnostics['prediction2']['mean2'])
            },
            'prediction3': {
                'test_name': diagnostics['prediction3']['test_name'],
                'p_value': float(diagnostics['prediction3']['p_value']),
                'significant': bool(diagnostics['prediction3']['significant']),
                'rel_imp_trans': float(diagnostics['prediction3']['rel_imp_trans']),
                'rel_imp_stable': float(diagnostics['prediction3']['rel_imp_stable'])
            }
        }
    }

    with open(os.path.join(save_dir, 'results_detailed.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run full experimental pipeline for Z2 Seam-Gated RNN'
    )

    # Data parameters
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--obs_dim', type=int, default=4)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--p_switch', type=float, default=0.05)
    parser.add_argument('--window_radius', type=int, default=5)
    parser.add_argument('--history_length', type=int, default=10)
    parser.add_argument('--sigma_epsilon', type=float, default=0.01)
    parser.add_argument('--sigma_eta', type=float, default=0.05)

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.05)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("Z2 SEAM-GATED RNN EXPERIMENTAL PIPELINE")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Set seed
    set_seed(args.seed)

    # Convert args to config dict
    config = vars(args)

    # 1. Generate data
    dataset = generate_data(config)

    # 2. Train neural models
    models = train_all_models(dataset, config, device)

    # 3. Create Hamilton/IMM filter
    hamilton_filter = create_hamilton_imm_filter(dataset, config)

    # 4. Evaluate all models
    results = evaluate_all_models(models, hamilton_filter, dataset, config, device)

    # 5. Run diagnostics
    diagnostics = run_diagnostics(models, results, dataset, config, device)

    # 6. Generate visualizations
    generate_visualizations(
        models, results, diagnostics, dataset, config,
        save_dir=os.path.join(args.save_dir, 'figures')
    )

    # 7. Save results
    save_results(results, diagnostics, config, args.save_dir)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
