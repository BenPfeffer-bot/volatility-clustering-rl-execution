# src/backtesting/performance/metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List
from ..strategies.base import Trade


class PerformanceMetrics:
    """
    Calculates key performance metrics for backtesting analysis.
    Focuses on institutional trading metrics from main.mdc:
    - Sharpe Ratio > 2.5
    - Win Rate > 68%
    - Max Drawdown < 4%
    - Average Return 0.75% per trade
    """

    def __init__(self, trades: List[Trade], portfolio_values: pd.Series):
        self.trades = trades
        self.portfolio_values = portfolio_values
        self.returns = portfolio_values.pct_change().dropna()

    def calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0

        # Handle NaN and inf values
        returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 2:
            return 0.0

        std = returns.std()
        if std == 0:
            return 0.0

        # Annualize using sqrt(252) for daily data
        return np.sqrt(252) * returns.mean() / std

    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio using only downside volatility."""
        if len(self.returns) < 2:
            return 0.0

        # Handle NaN and inf values
        returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return (
                0.0 if returns.mean() <= 0 else 100.0
            )  # Large but finite number for all positive returns

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        return np.sqrt(252) * returns.mean() / downside_std

    def calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        if not self.trades:
            return 0.0

        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum peak to trough drawdown."""
        if len(self.portfolio_values) < 2:
            return 0.0

        peak = self.portfolio_values.expanding().max()
        drawdown = (self.portfolio_values - peak) / peak
        return abs(drawdown.min())

    def calculate_avg_trade_return(self) -> float:
        """Calculate average return per trade."""
        if not self.trades:
            return 0.0

        returns = [
            trade.pnl / trade.initial_value
            for trade in self.trades
            if trade.initial_value != 0
        ]
        if not returns:
            return 0.0

        # Remove any inf values
        returns = [r for r in returns if not np.isinf(r) and not np.isnan(r)]
        return np.mean(returns) if returns else 0.0

    def calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes."""
        if not self.trades:
            return 0.0

        durations = [trade.holding_time for trade in self.trades if trade.holding_time]
        return np.mean(durations) if durations else 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate ratio of gross profits to gross losses."""
        if not self.trades:
            return 0.0

        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))

        return gross_profit / gross_loss if gross_loss != 0 else float("inf")

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        return {
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "win_rate": self.calculate_win_rate(),
            "max_drawdown": self.calculate_max_drawdown(),
            "avg_trade_return": self.calculate_avg_trade_return(),
            "avg_trade_duration": self.calculate_avg_trade_duration(),
            "profit_factor": self.calculate_profit_factor(),
            "total_trades": len(self.trades),
            "total_return": (
                self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]
            )
            - 1
            if len(self.portfolio_values) > 1
            else 0.0,
        }

    def check_performance_targets(self) -> Dict[str, bool]:
        """Check if performance meets targets from main.mdc."""
        metrics = self.calculate_all_metrics()

        return {
            "meets_sharpe_target": metrics["sharpe_ratio"] > 2.5,
            "meets_win_rate_target": metrics["win_rate"] > 0.68,
            "meets_drawdown_target": metrics["max_drawdown"] < 0.04,
            "meets_return_target": metrics["avg_trade_return"] > 0.0075,
            "meets_duration_target": 10 <= metrics["avg_trade_duration"] <= 30,
        }

    def generate_summary_report(self) -> str:
        """Generate detailed performance summary report."""
        metrics = self.calculate_all_metrics()
        targets = self.check_performance_targets()

        report = []
        report.append("Performance Summary Report")
        report.append("=" * 50)

        # Core metrics
        report.append("\nCore Performance Metrics:")
        report.append(f"Total Return: {metrics['total_return']:.2%}")
        report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        report.append(f"Win Rate: {metrics['win_rate']:.2%}")
        report.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        report.append(f"Average Trade Return: {metrics['avg_trade_return']:.2%}")
        report.append(
            f"Average Trade Duration: {metrics['avg_trade_duration']:.1f} minutes"
        )
        report.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
        report.append(f"Total Trades: {metrics['total_trades']}")

        # Target achievement
        report.append("\nTarget Achievement:")
        report.append(
            f"✓ Sharpe Ratio > 2.5: {'Yes' if targets['meets_sharpe_target'] else 'No'}"
        )
        report.append(
            f"✓ Win Rate > 68%: {'Yes' if targets['meets_win_rate_target'] else 'No'}"
        )
        report.append(
            f"✓ Max Drawdown < 4%: {'Yes' if targets['meets_drawdown_target'] else 'No'}"
        )
        report.append(
            f"✓ Avg Return > 0.75%: {'Yes' if targets['meets_return_target'] else 'No'}"
        )
        report.append(
            f"✓ Duration 10-30 min: {'Yes' if targets['meets_duration_target'] else 'No'}"
        )

        return "\n".join(report)
