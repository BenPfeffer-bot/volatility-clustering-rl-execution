import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(returns, positions, trades):
    """Calculate performance metrics with proper validation."""
    metrics = {}

    # Validate inputs
    if len(returns) == 0:
        logger.warning("Empty returns array, skipping metrics calculation")
        return metrics

    # Calculate returns-based metrics
    metrics["total_return"] = np.sum(returns)

    # Handle empty arrays and zeros for volatility calculation
    if len(returns) > 1:
        vol = np.std(returns, ddof=1)
        metrics["volatility"] = vol if vol != 0 else np.finfo(float).eps
        metrics["sharpe_ratio"] = (
            np.mean(returns) / metrics["volatility"] if metrics["volatility"] > 0 else 0
        )
    else:
        metrics["volatility"] = 0
        metrics["sharpe_ratio"] = 0

    # Calculate trade-based metrics
    if trades is not None and len(trades) > 0:
        winning_trades = np.sum(trades > 0)
        metrics["win_rate"] = winning_trades / len(trades) if len(trades) > 0 else 0
        metrics["total_trades"] = len(trades)
    else:
        metrics["win_rate"] = 0
        metrics["total_trades"] = 0

    # Calculate position-based metrics
    if positions is not None and len(positions) > 0:
        metrics["avg_position_size"] = np.mean(np.abs(positions))
        metrics["max_position"] = np.max(np.abs(positions))
    else:
        metrics["avg_position_size"] = 0
        metrics["max_position"] = 0

    return metrics
