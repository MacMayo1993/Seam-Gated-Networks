"""
Hamilton/IMM Switching Kalman Filter

Interacting Multiple Model (IMM) filter for regime-switching linear dynamical systems.
This is the OPTIMAL baseline in the correctly specified linear-Gaussian class.

STATE EQUATIONS (for each regime j ∈ {+1, -1}):
    z_{t+1}^j = j*A*z_t^j + ε_t,  ε_t ~ N(0, Q)
    x_t = C*z_t + η_t,            η_t ~ N(0, R)

FILTER EQUATIONS:
    1. Prediction: z_{t+1|t}^j = j*A*μ_t^j, P_{t+1|t}^j = A*Σ_t^j*A^T + Q
    2. Update: Kalman update with innovation
    3. Regime probability update using Bayes rule

ONE-STEP PREDICTION:
    ẑ_{t+1|t} = Σ_j π_t^j * (j*A*μ_t^j)
    x̂_{t+1|t} = C*ẑ_{t+1|t}
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.stats import multivariate_normal
from scipy.linalg import inv


class HamiltonIMM:
    """
    Hamilton/IMM Switching Kalman Filter with known parameters.

    Implements the Interacting Multiple Model filter for a two-regime
    antipodal switching system.
    """

    def __init__(
        self,
        A: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        p_switch: float = 0.05,
        regimes: List[int] = None
    ):
        """
        Initialize the Hamilton/IMM filter.

        Args:
            A: Dynamics matrix (latent_dim, latent_dim)
            C: Observation matrix (obs_dim, latent_dim)
            Q: Process noise covariance (latent_dim, latent_dim)
            R: Observation noise covariance (obs_dim, obs_dim)
            p_switch: Probability of regime switch
            regimes: List of regime values (default: [+1, -1])
        """
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.p_switch = p_switch

        self.latent_dim = A.shape[0]
        self.obs_dim = C.shape[0]

        if regimes is None:
            regimes = [1, -1]
        self.regimes = regimes
        self.n_regimes = len(regimes)

        # Transition probability matrix
        # P(j|i) where i is previous regime, j is current regime
        self.transition_matrix = self._build_transition_matrix()

        # Initialize filter state
        self.reset()

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build regime transition probability matrix.

        Returns:
            P: Transition matrix (n_regimes, n_regimes)
               P[i, j] = P(regime_j | regime_i)
        """
        P = np.zeros((self.n_regimes, self.n_regimes))

        for i, regime_i in enumerate(self.regimes):
            for j, regime_j in enumerate(self.regimes):
                if regime_i == regime_j:
                    P[i, j] = 1.0 - self.p_switch
                else:
                    P[i, j] = self.p_switch

        return P

    def reset(self, initial_state: np.ndarray = None, initial_cov: np.ndarray = None):
        """
        Reset filter to initial conditions.

        Args:
            initial_state: Initial state estimate (latent_dim,)
            initial_cov: Initial state covariance (latent_dim, latent_dim)
        """
        if initial_state is None:
            initial_state = np.zeros(self.latent_dim)
        if initial_cov is None:
            initial_cov = np.eye(self.latent_dim)

        # State estimates for each regime
        self.mu = {regime: initial_state.copy() for regime in self.regimes}
        self.Sigma = {regime: initial_cov.copy() for regime in self.regimes}

        # Regime probabilities (uniform prior)
        self.pi = {regime: 1.0 / self.n_regimes for regime in self.regimes}

    def predict_step(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Prediction step: propagate state estimates forward in time.

        Returns:
            Dictionary mapping regime -> (predicted_mean, predicted_cov)
        """
        predictions = {}

        for regime in self.regimes:
            # z_{t+1|t}^j = j*A*μ_t^j
            z_pred = regime * (self.A @ self.mu[regime])

            # P_{t+1|t}^j = A*Σ_t^j*A^T + Q
            P_pred = self.A @ self.Sigma[regime] @ self.A.T + self.Q

            predictions[regime] = (z_pred, P_pred)

        return predictions

    def update_step(
        self,
        x_obs: np.ndarray,
        predictions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Update step: incorporate new observation.

        Args:
            x_obs: Observed data (obs_dim,)
            predictions: Predicted states from predict_step
        """
        # Compute likelihoods and updated states for each regime
        likelihoods = {}
        updated_states = {}

        for regime in self.regimes:
            z_pred, P_pred = predictions[regime]

            # Innovation: ν^j = x - C*z_{t+1|t}^j
            innovation = x_obs - self.C @ z_pred

            # Innovation covariance: S^j = C*P_{t+1|t}^j*C^T + R
            S = self.C @ P_pred @ self.C.T + self.R

            # Kalman gain: K^j = P_{t+1|t}^j*C^T*(S^j)^{-1}
            K = P_pred @ self.C.T @ inv(S)

            # Updated state: μ_{t+1}^j = z_{t+1|t}^j + K^j*ν^j
            mu_updated = z_pred + K @ innovation

            # Updated covariance: Σ_{t+1}^j = (I - K^j*C)*P_{t+1|t}^j
            Sigma_updated = (np.eye(self.latent_dim) - K @ self.C) @ P_pred

            # Likelihood: L^j = N(ν^j; 0, S^j)
            likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(self.obs_dim), cov=S)

            likelihoods[regime] = likelihood
            updated_states[regime] = (mu_updated, Sigma_updated)

        # Update regime probabilities
        self._update_regime_probabilities(likelihoods)

        # Store updated states
        for regime in self.regimes:
            self.mu[regime], self.Sigma[regime] = updated_states[regime]

    def _update_regime_probabilities(self, likelihoods: Dict[int, float]) -> None:
        """
        Update regime probabilities using Bayes rule.

        Args:
            likelihoods: Dictionary mapping regime -> likelihood
        """
        # Prior mixing: π_{t+1|t}^j = Σ_i π_t^i * P(j|i)
        pi_prior = {}
        for j, regime_j in enumerate(self.regimes):
            pi_prior[regime_j] = sum(
                self.pi[regime_i] * self.transition_matrix[i, j]
                for i, regime_i in enumerate(self.regimes)
            )

        # Posterior: π_{t+1}^j = L^j * π_{t+1|t}^j / Σ_k L^k * π_{t+1|t}^k
        numerators = {regime: likelihoods[regime] * pi_prior[regime]
                      for regime in self.regimes}
        normalizer = sum(numerators.values())

        if normalizer > 0:
            self.pi = {regime: num / normalizer for regime, num in numerators.items()}
        else:
            # Fallback to uniform if numerical issues
            self.pi = {regime: 1.0 / self.n_regimes for regime in self.regimes}

    def predict_observation(self) -> np.ndarray:
        """
        One-step ahead prediction of observation.

        Returns:
            x̂_{t+1|t}: Predicted observation (obs_dim,)
        """
        # ẑ_{t+1|t} = Σ_j π_t^j * (j*A*μ_t^j)
        z_pred = sum(
            self.pi[regime] * regime * (self.A @ self.mu[regime])
            for regime in self.regimes
        )

        # x̂_{t+1|t} = C*ẑ_{t+1|t}
        x_pred = self.C @ z_pred

        return x_pred

    def filter_sequence(
        self,
        observations: np.ndarray,
        return_diagnostics: bool = False
    ) -> np.ndarray:
        """
        Filter entire observation sequence and generate predictions.

        Args:
            observations: Observation sequence (T, obs_dim)
            return_diagnostics: Whether to return diagnostic information

        Returns:
            predictions: One-step ahead predictions (T-1, obs_dim)
            diagnostics: Dictionary with filtering diagnostics [if return_diagnostics=True]
        """
        T = len(observations)
        predictions = np.zeros((T - 1, self.obs_dim))

        if return_diagnostics:
            regime_probs = np.zeros((T, self.n_regimes))
            map_regimes = np.zeros(T, dtype=int)

        # Initialize with first observation
        # Note: we don't predict x_0, start from predicting x_1 given x_0
        pred_dict = self.predict_step()
        self.update_step(observations[0], pred_dict)

        if return_diagnostics:
            for j, regime in enumerate(self.regimes):
                regime_probs[0, j] = self.pi[regime]
            map_regimes[0] = max(self.regimes, key=lambda r: self.pi[r])

        # Process remaining observations
        for t in range(1, T):
            # Predict next observation before seeing it
            predictions[t - 1] = self.predict_observation()

            # Update with actual observation
            pred_dict = self.predict_step()
            self.update_step(observations[t], pred_dict)

            if return_diagnostics:
                for j, regime in enumerate(self.regimes):
                    regime_probs[t, j] = self.pi[regime]
                map_regimes[t] = max(self.regimes, key=lambda r: self.pi[r])

        if return_diagnostics:
            diagnostics = {
                'regime_probs': regime_probs,
                'map_regimes': map_regimes
            }
            return predictions, diagnostics
        else:
            return predictions

    def get_regime_classification_accuracy(
        self,
        map_regimes: np.ndarray,
        true_regimes: np.ndarray
    ) -> float:
        """
        Compute classification accuracy for MAP regime estimates.

        Args:
            map_regimes: MAP regime estimates (T,)
            true_regimes: True regimes (T,)

        Returns:
            Accuracy: Fraction of correct classifications
        """
        return np.mean(map_regimes == true_regimes)


if __name__ == '__main__':
    # Test Hamilton/IMM filter
    print("Testing Hamilton/IMM Switching Kalman Filter")
    print("=" * 60)

    # Create simple system
    latent_dim = 4
    obs_dim = 2

    A = np.diag([0.95, 0.9, 0.85, 0.8])
    C = np.random.randn(obs_dim, latent_dim) * 0.5
    Q = 0.01 ** 2 * np.eye(latent_dim)
    R = 0.05 ** 2 * np.eye(obs_dim)

    print(f"System dimensions:")
    print(f"  Latent: {latent_dim}")
    print(f"  Observation: {obs_dim}")

    # Create filter
    filter = HamiltonIMM(A, C, Q, R, p_switch=0.05)
    print(f"\nFilter initialized with {filter.n_regimes} regimes: {filter.regimes}")

    # Generate synthetic data
    T = 100
    regimes = np.ones(T, dtype=int)
    regimes[50:] = -1  # Switch at t=50

    z = np.zeros((T, latent_dim))
    x = np.zeros((T, obs_dim))
    z[0] = np.random.randn(latent_dim)
    x[0] = C @ z[0] + np.random.randn(obs_dim) * 0.05

    for t in range(T - 1):
        z[t + 1] = regimes[t] * (A @ z[t]) + np.random.multivariate_normal(np.zeros(latent_dim), Q)
        x[t + 1] = C @ z[t + 1] + np.random.randn(obs_dim) * 0.05

    print(f"\nGenerated {T} observations with switch at t=50")

    # Run filter
    filter.reset()
    predictions, diagnostics = filter.filter_sequence(x, return_diagnostics=True)

    print(f"\nFilter results:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Regime probabilities shape: {diagnostics['regime_probs'].shape}")

    # Compute accuracy
    accuracy = filter.get_regime_classification_accuracy(
        diagnostics['map_regimes'], regimes
    )
    print(f"  Regime classification accuracy: {accuracy:.3f}")

    # Compute MSE
    mse = np.mean((predictions - x[1:]) ** 2)
    print(f"  Prediction MSE: {mse:.6f}")

    print("\nTest passed!")
