"""
Core metrics calculation for backtesting performance tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class BacktestMetrics:
    """
    Core metrics calculation for backtesting that tracks:
    - Portfolio performance
    - Risk metrics
    - Trading statistics
    """

    def __init__(self):
        self.metrics_history: Dict[str, pd.Series] = {}

    def calculate_metrics(
        self, trades: List[Trade], portfolio_values: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Args:
            trades: List of completed trades
            portfolio_values: Series of portfolio values

        Returns:
            Dict of performance metrics
        """
        if not trades or len(portfolio_values) < 2:
            return self._get_empty_metrics()

        # Calculate returns
        returns = portfolio_values.pct_change().dropna()

        # Performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Trading metrics
        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        avg_return = np.mean([t.pnl / t.initial_value for t in trades])
        avg_win = (
            np.mean([t.pnl / t.initial_value for t in trades if t.pnl > 0])
            if any(t.pnl > 0 for t in trades)
            else 0
        )
        avg_loss = (
            np.mean([t.pnl / t.initial_value for t in trades if t.pnl < 0])
            if any(t.pnl < 0 for t in trades)
            else 0
        )

        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        volatility = returns.std() * np.sqrt(252)

        # Trade statistics
        durations = [t.holding_time for t in trades if t.holding_time]
        avg_duration = np.mean(durations) if durations else 0

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "avg_trade_return": avg_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "avg_duration": avg_duration,
            "total_trades": len(trades),
            "profit_factor": self._calculate_profit_factor(trades),
        }

        # Store metrics history
        self.metrics_history[pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")] = pd.Series(
            metrics
        )

        return metrics

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float("inf")

        downside_std = downside_returns.std()
        return (
            np.sqrt(252) * returns.mean() / downside_std if downside_std != 0 else 0.0
        )

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum peak to trough drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())

    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate ratio of gross profits to gross losses."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float("inf")

    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "avg_duration": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
        }

    def generate_metrics_report(self) -> str:
        """Generate detailed metrics report."""
        if not self.metrics_history:
            return "No metrics history available"

        latest_metrics = list(self.metrics_history.values())[-1]

        report = []
        report.append("Backtest Performance Report")
        report.append("=" * 50)

        # Performance metrics
        report.append("\nPerformance Metrics:")
        report.append(f"Total Return: {latest_metrics['total_return']:.2%}")
        report.append(f"Sharpe Ratio: {latest_metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {latest_metrics['sortino_ratio']:.2f}")

        # Trading metrics
        report.append("\nTrading Metrics:")
        report.append(f"Total Trades: {latest_metrics['total_trades']}")
        report.append(f"Win Rate: {latest_metrics['win_rate']:.2%}")
        report.append(f"Average Trade Return: {latest_metrics['avg_trade_return']:.2%}")
        report.append(f"Average Win: {latest_metrics['avg_win']:.2%}")
        report.append(f"Average Loss: {latest_metrics['avg_loss']:.2%}")

        # Risk metrics
        report.append("\nRisk Metrics:")
        report.append(f"Maximum Drawdown: {latest_metrics['max_drawdown']:.2%}")
        report.append(f"Annualized Volatility: {latest_metrics['volatility']:.2%}")
        report.append(f"Profit Factor: {latest_metrics['profit_factor']:.2f}")
        report.append(f"Average Duration: {latest_metrics['avg_duration']:.1f} minutes")

        return "\n".join(report)
