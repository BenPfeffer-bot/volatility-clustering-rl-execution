"""
Bayesian optimization for strategy parameter tuning.
Uses Gaussian Process to optimize strategy parameters based on performance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameters using Gaussian Process.
    Optimizes for multiple objectives:
    - Sharpe Ratio
    - Win Rate
    - Average Return
    - Maximum Drawdown
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_iterations: int = 50,
        exploration_weight: float = 0.1,
    ):
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight

        # Adjust kernel parameters for better convergence
        n_params = len(param_bounds)
        length_scales = [10.0] * n_params  # More reasonable starting length scales
        kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) * RBF(
            length_scales,
            length_scale_bounds=[(0.1, 1000.0)] * n_params,  # Reduced upper bound
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,  # Increased restarts
            random_state=42,
            normalize_y=True,  # Add normalization
            alpha=1e-6,  # Add small noise
        )
        self.X_samples = []
        self.y_samples = []

    def _objective_function(
        self,
        params: Dict[str, float],
        evaluate_strategy: Callable,
    ) -> float:
        """
        Calculate composite objective value from strategy performance.

        Args:
            params: Strategy parameters
            evaluate_strategy: Function to evaluate strategy with parameters

        Returns:
            Composite score balancing multiple objectives
        """
        # Evaluate strategy with parameters
        metrics = evaluate_strategy(params)

        # Extract key metrics with better error handling
        sharpe = float(metrics.get("sharpe_ratio", 0))
        win_rate = float(metrics.get("win_rate", 0))
        avg_return = float(metrics.get("avg_trade_return", 0))
        max_dd = float(metrics.get("max_drawdown", 1))

        # Handle invalid values
        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = 0.0
        if np.isnan(win_rate) or np.isinf(win_rate):
            win_rate = 0.0
        if np.isnan(avg_return) or np.isinf(avg_return):
            avg_return = 0.0
        if np.isnan(max_dd) or np.isinf(max_dd) or max_dd > 1:
            max_dd = 1.0

        # Calculate composite score
        score = (
            2.0 * min(sharpe, 10.0)  # Cap extreme values
            + 1.0 * win_rate
            + 1.0 * min(avg_return / 0.0075, 2.0)  # Cap normalized return
            + 1.0 * (1 - max_dd / 0.04)
        ) / 5.0

        return max(score, -10.0)  # Prevent extreme negative values

    def _sample_parameters(self) -> Dict[str, float]:
        """Sample random parameters within bounds."""
        params = {}
        for name, (low, high) in self.param_bounds.items():
            params[name] = np.random.uniform(low, high)
        return params

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters dict to array."""
        return np.array([params[name] for name in self.param_bounds.keys()])

    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert array to parameters dict."""
        return {name: float(x[i]) for i, name in enumerate(self.param_bounds.keys())}

    def _expected_improvement(
        self,
        x: np.ndarray,
        xi: float = 0.01,
    ) -> float:
        """
        Calculate expected improvement at point x.

        Args:
            x: Parameters to evaluate
            xi: Exploration-exploitation trade-off parameter

        Returns:
            Expected improvement value
        """
        mu, sigma = self.gp.predict(x.reshape(1, -1), return_std=True)
        mu = mu[0]
        sigma = sigma[0]

        mu_sample = np.max(self.y_samples)

        with np.errstate(divide="warn"):
            imp = mu - mu_sample - xi
            Z = imp / sigma if sigma > 0 else 0
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei = 0.0 if sigma == 0.0 else ei

        return ei

    def optimize(
        self,
        evaluate_strategy: Callable,
        n_random_starts: int = 10,
    ) -> Tuple[Dict[str, float], float]:
        """
        Run Bayesian optimization process.

        Args:
            evaluate_strategy: Function to evaluate strategy with parameters
            n_random_starts: Number of random evaluations before optimization

        Returns:
            Tuple of (best_parameters, best_score)
        """
        # Initial random sampling
        logger.info("Starting random sampling phase...")
        for _ in range(n_random_starts):
            params = self._sample_parameters()
            score = self._objective_function(params, evaluate_strategy)

            self.X_samples.append(self._params_to_array(params))
            self.y_samples.append(score)

        self.X_samples = np.array(self.X_samples)
        self.y_samples = np.array(self.y_samples)

        # Fit GP model
        self.gp.fit(self.X_samples, self.y_samples)

        # Optimization loop
        logger.info("Starting optimization phase...")
        for i in range(self.n_iterations):
            # Find point with highest expected improvement
            best_ei = 0
            best_params = None

            for _ in range(100):  # Sample 100 random points
                params = self._sample_parameters()
                x = self._params_to_array(params)
                ei = self._expected_improvement(x)

                if ei > best_ei:
                    best_ei = ei
                    best_params = params

            if best_params is None:
                logger.warning("No improvement found, stopping optimization")
                break

            # Evaluate point
            score = self._objective_function(best_params, evaluate_strategy)

            # Update samples
            self.X_samples = np.vstack(
                (self.X_samples, self._params_to_array(best_params))
            )
            self.y_samples = np.append(self.y_samples, score)

            # Update model
            self.gp.fit(self.X_samples, self.y_samples)

            logger.info(
                f"Iteration {i + 1}/{self.n_iterations}, "
                f"Best score: {np.max(self.y_samples):.4f}"
            )

        # Get best parameters
        best_idx = np.argmax(self.y_samples)
        best_params = self._array_to_params(self.X_samples[best_idx])
        best_score = self.y_samples[best_idx]

        return best_params, best_score

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.y_samples)), self.y_samples, "b-", label="Score")
            plt.plot(
                range(len(self.y_samples)),
                np.maximum.accumulate(self.y_samples),
                "r--",
                label="Best score",
            )
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.title("Optimization History")
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
