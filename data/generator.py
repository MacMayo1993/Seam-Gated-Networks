"""
Antipodal Regime Switching Data Generator

Generates time series from a regime-switching linear dynamical system where
the regime affects the sign of the dynamics matrix.

Mathematical specification:
    r_t ∈ {+1, -1}
    P(r_{t+1} = -r_t) = p_switch
    P(r_{t+1} = r_t) = 1 - p_switch

    z_{t+1} = r_t * A * z_t + ε_t,  ε_t ~ N(0, Q)
    x_t = C * z_t + η_t,            η_t ~ N(0, R)

where:
    - A: dynamics matrix (diagonalizable with specified eigenvalues)
    - C: observation matrix
    - Q: process noise covariance
    - R: observation noise covariance
"""

import numpy as np
from typing import Tuple, Dict
from scipy.linalg import qr


class AntipodalRegimeSwitchingGenerator:
    """Generate time series from antipodal regime switching system."""

    def __init__(
        self,
        latent_dim: int = 8,
        obs_dim: int = 4,
        p_switch: float = 0.05,
        eigenvalues: np.ndarray = None,
        sigma_epsilon: float = 0.01,
        sigma_eta: float = 0.05,
        seed: int = None
    ):
        """
        Initialize the data generator.

        Args:
            latent_dim: Dimension of latent state z_t
            obs_dim: Dimension of observations x_t
            p_switch: Probability of regime switch
            eigenvalues: Eigenvalues for dynamics matrix A (default: canonical values)
            sigma_epsilon: Process noise standard deviation
            sigma_eta: Observation noise standard deviation
            seed: Random seed for reproducibility
        """
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.p_switch = p_switch
        self.sigma_epsilon = sigma_epsilon
        self.sigma_eta = sigma_eta

        if seed is not None:
            np.random.seed(seed)

        # Canonical eigenvalues if not provided
        if eigenvalues is None:
            eigenvalues = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6])

        self.eigenvalues = eigenvalues

        # Construct dynamics matrix A as Q @ diag(eigenvalues) @ Q^T
        # with random orthogonal matrix Q
        random_matrix = np.random.randn(latent_dim, latent_dim)
        Q, _ = qr(random_matrix)  # QR decomposition gives orthogonal Q

        self.A = Q @ np.diag(eigenvalues) @ Q.T

        # Construct observation matrix C with entries ~ N(0, 1/4)
        self.C = np.random.randn(obs_dim, latent_dim) * 0.5

        # Covariance matrices
        self.Q = (sigma_epsilon ** 2) * np.eye(latent_dim)
        self.R = (sigma_eta ** 2) * np.eye(obs_dim)

    def generate(
        self,
        T: int,
        initial_regime: int = 1,
        return_latent: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate a time series of length T.

        Args:
            T: Length of time series
            initial_regime: Initial regime r_0 ∈ {+1, -1}
            return_latent: Whether to return latent states z_t

        Returns:
            Dictionary containing:
                - x: Observations (T, obs_dim)
                - regimes: Regime sequence (T,)
                - z: Latent states (T, latent_dim) [if return_latent=True]
        """
        # Initialize
        regimes = np.zeros(T, dtype=int)
        regimes[0] = initial_regime

        z = np.zeros((T, self.latent_dim))
        x = np.zeros((T, self.obs_dim))

        # Initial state: z_0 ~ N(0, I)
        z[0] = np.random.randn(self.latent_dim)

        # Initial observation
        x[0] = self.C @ z[0] + np.random.randn(self.obs_dim) * self.sigma_eta

        # Generate time series
        for t in range(T - 1):
            # Sample regime for next timestep
            if np.random.rand() < self.p_switch:
                regimes[t + 1] = -regimes[t]
            else:
                regimes[t + 1] = regimes[t]

            # Evolve latent state: z_{t+1} = r_t * A * z_t + ε_t
            process_noise = np.random.multivariate_normal(
                np.zeros(self.latent_dim), self.Q
            )
            z[t + 1] = regimes[t] * (self.A @ z[t]) + process_noise

            # Generate observation: x_{t+1} = C * z_{t+1} + η_{t+1}
            obs_noise = np.random.randn(self.obs_dim) * self.sigma_eta
            x[t + 1] = self.C @ z[t + 1] + obs_noise

        result = {
            'x': x,
            'regimes': regimes
        }

        if return_latent:
            result['z'] = z

        return result

    def get_system_matrices(self) -> Dict[str, np.ndarray]:
        """
        Get the system matrices.

        Returns:
            Dictionary with keys: A, C, Q, R
        """
        return {
            'A': self.A,
            'C': self.C,
            'Q': self.Q,
            'R': self.R
        }


def create_supervised_dataset(
    x: np.ndarray,
    history_length: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised learning dataset from time series.

    Args:
        x: Time series data (T, obs_dim)
        history_length: Number of past observations to use as input

    Returns:
        X: Input sequences (N, history_length, obs_dim)
        y: Target observations (N, obs_dim)

    where N = T - history_length
    """
    T, obs_dim = x.shape
    N = T - history_length

    X = np.zeros((N, history_length, obs_dim))
    y = np.zeros((N, obs_dim))

    for i in range(N):
        X[i] = x[i:i + history_length]
        y[i] = x[i + history_length]

    return X, y


if __name__ == '__main__':
    # Test the generator
    print("Testing Antipodal Regime Switching Generator")
    print("=" * 60)

    generator = AntipodalRegimeSwitchingGenerator(seed=42)
    data = generator.generate(T=1000, return_latent=True)

    print(f"Generated time series with T={len(data['x'])} timesteps")
    print(f"Observation dimension: {data['x'].shape[1]}")
    print(f"Latent dimension: {data['z'].shape[1]}")
    print(f"\nRegime statistics:")
    print(f"  +1 regime: {np.sum(data['regimes'] == 1)} timesteps")
    print(f"  -1 regime: {np.sum(data['regimes'] == -1)} timesteps")

    # Count regime switches
    switches = np.sum(np.diff(data['regimes']) != 0)
    print(f"  Total switches: {switches}")
    print(f"  Empirical switch rate: {switches / len(data['regimes']):.4f}")
    print(f"  Expected switch rate: {generator.p_switch:.4f}")

    # System matrices
    matrices = generator.get_system_matrices()
    print(f"\nSystem matrix shapes:")
    for name, mat in matrices.items():
        print(f"  {name}: {mat.shape}")

    # Create supervised dataset
    X, y = create_supervised_dataset(data['x'], history_length=10)
    print(f"\nSupervised dataset:")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
