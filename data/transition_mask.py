"""
Transition Window Mask Computation

Computes binary masks indicating which prediction indices fall within
transition windows around regime switches.

TRANSITION WINDOW DEFINITION:
A prediction index t is "transition" if:
    ∃s : r_s ≠ r_{s-1} AND t ∈ [s-w, s+w-1]

All other indices are "stable"
"""

import numpy as np
from typing import Tuple


def compute_transition_mask(
    regimes: np.ndarray,
    window_radius: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transition window mask for a regime sequence.

    Args:
        regimes: Regime sequence (T,) with values in {+1, -1}
        window_radius: Radius w of transition window

    Returns:
        transition_mask: Binary array (T,) where 1 = transition, 0 = stable
        switch_indices: Array of indices where regime switches occur
    """
    T = len(regimes)

    # Find regime switches: indices where r_s ≠ r_{s-1}
    # Note: switches[s] = True means switch from s-1 to s
    switches = np.diff(regimes) != 0
    switch_indices = np.where(switches)[0] + 1  # +1 because diff reduces length by 1

    # Initialize mask as all stable (0)
    transition_mask = np.zeros(T, dtype=bool)

    # Mark transition windows around each switch
    for s in switch_indices:
        # Window: [s - w, s + w - 1]
        window_start = max(0, s - window_radius)
        window_end = min(T, s + window_radius)

        transition_mask[window_start:window_end] = True

    return transition_mask.astype(int), switch_indices


def compute_transition_statistics(
    regimes: np.ndarray,
    transition_mask: np.ndarray
) -> dict:
    """
    Compute statistics about transitions in the data.

    Args:
        regimes: Regime sequence (T,)
        transition_mask: Binary transition mask (T,)

    Returns:
        Dictionary with statistics
    """
    T = len(regimes)
    n_switches = np.sum(np.diff(regimes) != 0)
    n_transition = np.sum(transition_mask)
    n_stable = T - n_transition

    stats = {
        'total_timesteps': T,
        'n_switches': n_switches,
        'n_transition': n_transition,
        'n_stable': n_stable,
        'frac_transition': n_transition / T,
        'frac_stable': n_stable / T,
        'empirical_switch_rate': n_switches / T
    }

    return stats


def split_indices_by_mask(
    indices: np.ndarray,
    transition_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split indices into transition and stable subsets.

    Args:
        indices: Array of indices to split
        transition_mask: Binary transition mask

    Returns:
        transition_indices: Subset of indices in transition windows
        stable_indices: Subset of indices in stable windows
    """
    transition_indices = indices[transition_mask[indices] == 1]
    stable_indices = indices[transition_mask[indices] == 0]

    return transition_indices, stable_indices


def apply_mask_to_data(
    data: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Apply binary mask to select subset of data.

    Args:
        data: Data array (N, ...)
        mask: Binary mask (N,)

    Returns:
        Subset of data where mask == 1
    """
    return data[mask == 1]


if __name__ == '__main__':
    # Test transition mask computation
    print("Testing Transition Mask Computation")
    print("=" * 60)

    # Create synthetic regime sequence with known switches
    T = 100
    regimes = np.ones(T, dtype=int)
    regimes[25:50] = -1  # Switch at t=25
    regimes[75:] = -1    # Switch at t=75

    print(f"Regime sequence length: {T}")
    print(f"Switches at indices: 25, 75")

    # Compute transition mask with different window radii
    for w in [0, 1, 5, 10]:
        mask, switches = compute_transition_mask(regimes, window_radius=w)
        stats = compute_transition_statistics(regimes, mask)

        print(f"\nWindow radius w={w}:")
        print(f"  Detected switches: {switches}")
        print(f"  Transition indices: {np.sum(mask)}")
        print(f"  Stable indices: {stats['n_stable']}")
        print(f"  Fraction transition: {stats['frac_transition']:.3f}")

    # Test splitting indices
    print("\n" + "=" * 60)
    print("Testing index splitting")

    w = 5
    mask, _ = compute_transition_mask(regimes, window_radius=w)

    # Indices for supervised learning (excluding first 10 for history)
    all_indices = np.arange(10, T)
    trans_idx, stable_idx = split_indices_by_mask(all_indices, mask)

    print(f"Window radius: {w}")
    print(f"Total prediction indices: {len(all_indices)}")
    print(f"Transition indices: {len(trans_idx)}")
    print(f"Stable indices: {len(stable_idx)}")
    print(f"Sum check: {len(trans_idx) + len(stable_idx)} == {len(all_indices)}")
