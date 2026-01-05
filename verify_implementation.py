"""
Verification Script for Mathematical Properties

Tests all critical mathematical properties of the implementation
without running the full experimental pipeline.
"""

import torch
import numpy as np
import sys

# Add current directory to path
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 80)
    print("TESTING IMPORTS")
    print("=" * 80)

    try:
        from data.generator import AntipodalRegimeSwitchingGenerator
        from data.transition_mask import compute_transition_mask
        from models.lstm import StandardLSTM
        from models.hamilton_imm import HamiltonIMM
        from models.z2_equivariant import Z2EquivariantRNN
        from models.z2_seam_gated import Z2SeamGatedRNN
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_z2_equivariant_properties():
    """Test Z2-equivariant model mathematical properties."""
    print("\n" + "=" * 80)
    print("TESTING Z2-EQUIVARIANT MODEL PROPERTIES")
    print("=" * 80)

    from models.z2_equivariant import Z2EquivariantRNN

    try:
        # Create model
        model = Z2EquivariantRNN(
            input_dim=4,
            hidden_dim=64,
            verify_properties=True
        )
        print("✓ Z2-Equivariant model created and verified")
        return True
    except Exception as e:
        print(f"✗ Z2-Equivariant test failed: {e}")
        return False


def test_z2_seam_gated_properties():
    """Test Z2 Seam-Gated model mathematical properties."""
    print("\n" + "=" * 80)
    print("TESTING Z2 SEAM-GATED MODEL PROPERTIES")
    print("=" * 80)

    from models.z2_seam_gated import Z2SeamGatedRNN, compute_k_star

    try:
        # Test k* value
        k_star = compute_k_star()
        expected = 0.721347520444
        assert abs(k_star - expected) < 1e-9, f"k* = {k_star}, expected {expected}"
        print(f"✓ k* = {k_star:.12f} (verified)")

        # Create model
        model = Z2SeamGatedRNN(
            input_dim=4,
            hidden_dim=64,
            temperature=0.05,
            verify_properties=True
        )
        print("✓ Z2 Seam-Gated model created and verified")

        # Test scale invariance
        from models.z2_seam_gated import compute_parity_energy

        h = torch.randn(64)
        alpha1 = compute_parity_energy(h, model.P_minus)
        alpha2 = compute_parity_energy(5.0 * h, model.P_minus)

        assert torch.allclose(alpha1, alpha2, atol=1e-6), \
            f"Scale invariance violated: {alpha1} != {alpha2}"
        print(f"✓ Scale invariance verified: α_-(h) = α_-(5h) = {alpha1:.6f}")

        return True
    except Exception as e:
        print(f"✗ Z2 Seam-Gated test failed: {e}")
        return False


def test_data_generator():
    """Test data generation."""
    print("\n" + "=" * 80)
    print("TESTING DATA GENERATOR")
    print("=" * 80)

    from data.generator import AntipodalRegimeSwitchingGenerator
    from data.transition_mask import compute_transition_mask

    try:
        # Create generator
        generator = AntipodalRegimeSwitchingGenerator(
            latent_dim=8,
            obs_dim=4,
            p_switch=0.05,
            seed=42
        )

        # Generate data
        data = generator.generate(T=1000, return_latent=True)

        assert data['x'].shape == (1000, 4), "Observation shape mismatch"
        assert data['z'].shape == (1000, 8), "Latent shape mismatch"
        assert data['regimes'].shape == (1000,), "Regime shape mismatch"

        print(f"✓ Generated time series: T={len(data['x'])}")

        # Count switches
        n_switches = np.sum(np.diff(data['regimes']) != 0)
        empirical_rate = n_switches / len(data['regimes'])
        print(f"✓ Regime switches: {n_switches} (rate={empirical_rate:.4f})")

        # Compute transition mask
        mask, switches = compute_transition_mask(data['regimes'], window_radius=5)
        print(f"✓ Transition mask computed: {np.sum(mask)} transition timesteps")

        return True
    except Exception as e:
        print(f"✗ Data generator test failed: {e}")
        return False


def test_hamilton_imm():
    """Test Hamilton/IMM filter."""
    print("\n" + "=" * 80)
    print("TESTING HAMILTON/IMM FILTER")
    print("=" * 80)

    from models.hamilton_imm import HamiltonIMM

    try:
        # Create simple system
        latent_dim = 4
        obs_dim = 2

        A = np.diag([0.95, 0.9, 0.85, 0.8])
        C = np.random.randn(obs_dim, latent_dim) * 0.5
        Q = 0.01 ** 2 * np.eye(latent_dim)
        R = 0.05 ** 2 * np.eye(obs_dim)

        # Create filter
        filter_obj = HamiltonIMM(A, C, Q, R, p_switch=0.05)
        print(f"✓ Hamilton/IMM filter created with {filter_obj.n_regimes} regimes")

        # Generate test data
        T = 100
        regimes = np.ones(T, dtype=int)
        regimes[50:] = -1

        z = np.zeros((T, latent_dim))
        x = np.zeros((T, obs_dim))
        z[0] = np.random.randn(latent_dim)
        x[0] = C @ z[0]

        for t in range(T - 1):
            z[t + 1] = regimes[t] * (A @ z[t]) + np.random.multivariate_normal(np.zeros(latent_dim), Q)
            x[t + 1] = C @ z[t + 1] + np.random.randn(obs_dim) * 0.05

        # Run filter
        filter_obj.reset()
        predictions, diagnostics = filter_obj.filter_sequence(x, return_diagnostics=True)

        print(f"✓ Filter ran successfully: {predictions.shape[0]} predictions")

        # Check regime classification
        accuracy = filter_obj.get_regime_classification_accuracy(
            diagnostics['map_regimes'], regimes
        )
        print(f"✓ Regime classification accuracy: {accuracy:.3f}")

        return True
    except Exception as e:
        print(f"✗ Hamilton/IMM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("Z2 SEAM-GATED RNN IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Z2-Equivariant Properties", test_z2_equivariant_properties()))
    results.append(("Z2 Seam-Gated Properties", test_z2_seam_gated_properties()))
    results.append(("Data Generator", test_data_generator()))
    results.append(("Hamilton/IMM Filter", test_hamilton_imm()))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
