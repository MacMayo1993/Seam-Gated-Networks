"""
Statistical Tests for Falsifiable Diagnostic Predictions

Implements tests for:
1. Parity energy peaks at transitions
2. Selective gate activation
3. Boundary-specific performance improvement
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def welch_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alternative: str = 'greater',
    alpha: float = 0.01
) -> Dict:
    """
    Perform Welch's t-test (unequal variances).

    Args:
        sample1: First sample
        sample2: Second sample
        alternative: 'greater', 'less', or 'two-sided'
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Compute t-statistic and p-value
    t_stat, p_value = stats.ttest_ind(
        sample1,
        sample2,
        equal_var=False,
        alternative=alternative
    )

    # Compute effect size (Cohen's d)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    n1, n2 = len(sample1), len(sample2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'n1': n1,
        'n2': n2,
        'cohens_d': cohens_d
    }


def prediction1_parity_energy_peaks(
    parity_energies: np.ndarray,
    transition_mask: np.ndarray,
    alpha: float = 0.01
) -> Dict:
    """
    Test Prediction 1: Parity energy peaks at transitions.

    H0: α_-(t) has same distribution in transition vs stable windows
    H1: Mean(α_-(t) | t ∈ T_trans) > Mean(α_-(t) | t ∈ T_stable)

    Args:
        parity_energies: Parity energy time series (T,)
        transition_mask: Binary transition mask (T,)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Split by mask
    trans_energies = parity_energies[transition_mask == 1]
    stable_energies = parity_energies[transition_mask == 0]

    # Perform Welch's t-test (one-sided: trans > stable)
    result = welch_t_test(trans_energies, stable_energies, alternative='greater', alpha=alpha)

    result['test_name'] = 'Prediction 1: Parity Energy Peaks at Transitions'
    result['hypothesis'] = 'Mean(α_- | transition) > Mean(α_- | stable)'

    return result


def prediction2_selective_gate_activation(
    gate_activations: np.ndarray,
    transition_mask: np.ndarray,
    alpha: float = 0.01
) -> Dict:
    """
    Test Prediction 2: Selective gate activation.

    H0: g(t) has same distribution in transition vs stable windows
    H1: Mean(g(t) | t ∈ T_trans) > Mean(g(t) | t ∈ T_stable)

    Args:
        gate_activations: Gate activation time series (T,)
        transition_mask: Binary transition mask (T,)
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Split by mask
    trans_gates = gate_activations[transition_mask == 1]
    stable_gates = gate_activations[transition_mask == 0]

    # Perform Welch's t-test (one-sided: trans > stable)
    result = welch_t_test(trans_gates, stable_gates, alternative='greater', alpha=alpha)

    result['test_name'] = 'Prediction 2: Selective Gate Activation'
    result['hypothesis'] = 'Mean(g | transition) > Mean(g | stable)'

    return result


def prediction3_boundary_specific_performance(
    mse_trans_hamilton: float,
    mse_trans_seam: float,
    mse_stable_hamilton: float,
    mse_stable_seam: float,
    predictions_hamilton: np.ndarray,
    predictions_seam: np.ndarray,
    targets: np.ndarray,
    transition_mask: np.ndarray,
    test_indices: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_seed: int = None
) -> Dict:
    """
    Test Prediction 3: Boundary-specific performance improvement.

    H0: Relative improvement vs Hamilton/IMM is same in transition and stable
    H1: Relative improvement is greater in transition than stable

    Uses bootstrap resampling for statistical test.

    Args:
        mse_trans_hamilton: Hamilton/IMM MSE on transition windows
        mse_trans_seam: Seam model MSE on transition windows
        mse_stable_hamilton: Hamilton/IMM MSE on stable windows
        mse_stable_seam: Seam model MSE on stable windows
        predictions_hamilton: Hamilton/IMM predictions (N, obs_dim)
        predictions_seam: Seam model predictions (N, obs_dim)
        targets: Ground truth (N, obs_dim)
        transition_mask: Binary transition mask for full series
        test_indices: Indices corresponding to predictions
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with test results
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Compute observed relative improvements
    rel_imp_trans = (mse_trans_hamilton - mse_trans_seam) / mse_trans_hamilton
    rel_imp_stable = (mse_stable_hamilton - mse_stable_seam) / mse_stable_hamilton

    observed_diff = rel_imp_trans - rel_imp_stable

    # Get squared errors
    se_hamilton = np.sum((predictions_hamilton - targets) ** 2, axis=1)
    se_seam = np.sum((predictions_seam - targets) ** 2, axis=1)

    # Split by mask
    mask_subset = transition_mask[test_indices]
    trans_idx = mask_subset == 1
    stable_idx = mask_subset == 0

    # Bootstrap
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # Resample transition indices
        if np.sum(trans_idx) > 0:
            trans_resample = np.random.choice(
                np.where(trans_idx)[0],
                size=np.sum(trans_idx),
                replace=True
            )
            mse_trans_h_boot = np.mean(se_hamilton[trans_resample])
            mse_trans_s_boot = np.mean(se_seam[trans_resample])
            rel_imp_trans_boot = (mse_trans_h_boot - mse_trans_s_boot) / mse_trans_h_boot
        else:
            rel_imp_trans_boot = 0.0

        # Resample stable indices
        if np.sum(stable_idx) > 0:
            stable_resample = np.random.choice(
                np.where(stable_idx)[0],
                size=np.sum(stable_idx),
                replace=True
            )
            mse_stable_h_boot = np.mean(se_hamilton[stable_resample])
            mse_stable_s_boot = np.mean(se_seam[stable_resample])
            rel_imp_stable_boot = (mse_stable_h_boot - mse_stable_s_boot) / mse_stable_h_boot
        else:
            rel_imp_stable_boot = 0.0

        bootstrap_diffs.append(rel_imp_trans_boot - rel_imp_stable_boot)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute p-value (one-sided: observed_diff > 0)
    p_value = np.mean(bootstrap_diffs <= 0)

    # Confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        'test_name': 'Prediction 3: Boundary-Specific Performance',
        'hypothesis': 'Rel. improvement (trans) > Rel. improvement (stable)',
        'rel_imp_trans': rel_imp_trans,
        'rel_imp_stable': rel_imp_stable,
        'observed_diff': observed_diff,
        'bootstrap_mean': np.mean(bootstrap_diffs),
        'bootstrap_std': np.std(bootstrap_diffs),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_bootstrap': n_bootstrap
    }


def print_diagnostic_results(results: Dict) -> None:
    """
    Pretty print diagnostic test results.

    Args:
        results: Dictionary with test results
    """
    print("\n" + "=" * 80)
    print(f"TEST: {results['test_name']}")
    print(f"Hypothesis: {results['hypothesis']}")
    print("=" * 80)

    if 't_statistic' in results:
        # Welch's t-test results
        print(f"Group 1: mean={results['mean1']:.6f}, std={results['std1']:.6f}, n={results['n1']}")
        print(f"Group 2: mean={results['mean2']:.6f}, std={results['std2']:.6f}, n={results['n2']}")
        print(f"t-statistic: {results['t_statistic']:.4f}")
        print(f"p-value: {results['p_value']:.6f}")
        print(f"Cohen's d: {results['cohens_d']:.4f}")
    else:
        # Bootstrap results
        print(f"Relative improvement (transition): {results['rel_imp_trans']:.4f}")
        print(f"Relative improvement (stable): {results['rel_imp_stable']:.4f}")
        print(f"Observed difference: {results['observed_diff']:.4f}")
        print(f"Bootstrap mean: {results['bootstrap_mean']:.4f}")
        print(f"Bootstrap 95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"p-value: {results['p_value']:.6f}")

    print(f"\nSignificant at α={results['alpha']}: {'YES ✓' if results['significant'] else 'NO ✗'}")
    print("=" * 80)


if __name__ == '__main__':
    # Test statistical functions
    print("Testing Statistical Diagnostics")
    print("=" * 60)

    # Create synthetic data
    T = 1000

    # Transition mask
    transition_mask = np.zeros(T, dtype=int)
    transition_mask[200:250] = 1
    transition_mask[600:650] = 1

    # Parity energies (higher in transition)
    parity_energies = np.random.randn(T) * 0.1 + 0.3
    parity_energies[transition_mask == 1] += 0.5  # Increase in transition

    # Gate activations (higher in transition)
    gate_activations = np.random.randn(T) * 0.1 + 0.2
    gate_activations[transition_mask == 1] += 0.4  # Increase in transition

    print("Testing Prediction 1: Parity energy peaks at transitions")
    result1 = prediction1_parity_energy_peaks(parity_energies, transition_mask)
    print_diagnostic_results(result1)

    print("\nTesting Prediction 2: Selective gate activation")
    result2 = prediction2_selective_gate_activation(gate_activations, transition_mask)
    print_diagnostic_results(result2)

    print("\nTesting Prediction 3: Boundary-specific performance")
    # Create dummy predictions
    N = 500
    obs_dim = 4
    test_indices = np.arange(500, 1000)

    predictions_hamilton = np.random.randn(N, obs_dim)
    predictions_seam = predictions_hamilton + np.random.randn(N, obs_dim) * 0.1  # Slightly better
    targets = np.random.randn(N, obs_dim)

    # Compute MSEs (dummy values)
    mse_trans_hamilton = 0.8
    mse_trans_seam = 0.5  # Big improvement in transition
    mse_stable_hamilton = 0.4
    mse_stable_seam = 0.35  # Small improvement in stable

    result3 = prediction3_boundary_specific_performance(
        mse_trans_hamilton,
        mse_trans_seam,
        mse_stable_hamilton,
        mse_stable_seam,
        predictions_hamilton,
        predictions_seam,
        targets,
        transition_mask,
        test_indices,
        n_bootstrap=100,
        random_seed=42
    )
    print_diagnostic_results(result3)

    print("\nTest passed!")
