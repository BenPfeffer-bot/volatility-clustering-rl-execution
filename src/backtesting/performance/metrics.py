# src/backtesting/performance/metrics.py

import pandas as pd
import numpy as np
from typing import Dict, List
from ..strategies.base import Trade
import logging

logger = logging.getLogger(__name__)


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

        try:
            # Handle NaN and inf values
            returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) < 2:
                return 0.0

            std = returns.std()
            if std == 0 or np.isnan(std) or np.isinf(std):
                return 0.0

            mean_return = returns.mean()
            if np.isnan(mean_return) or np.isinf(mean_return):
                return 0.0

            # Annualize using sqrt(252) for daily data
            sharpe = np.sqrt(252) * mean_return / std
            return (
                float(sharpe) if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio using only downside volatility."""
        if len(self.returns) < 2:
            return 0.0

        try:
            # Handle NaN and inf values
            returns = self.returns.replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            if np.isnan(mean_return) or np.isinf(mean_return):
                return 0.0

            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 100.0 if mean_return > 0 else 0.0

            downside_std = np.std(downside_returns)
            if downside_std == 0 or np.isnan(downside_std) or np.isinf(downside_std):
                return 0.0

            sortino = np.sqrt(252) * mean_return / downside_std
            return (
                float(sortino)
                if not np.isnan(sortino) and not np.isinf(sortino)
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        if not self.trades:
            return 0.0

        try:
            winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
            win_rate = winning_trades / len(self.trades)
            return (
                float(win_rate)
                if not np.isnan(win_rate) and not np.isinf(win_rate)
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating win rate: {e}")
            return 0.0

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum peak to trough drawdown."""
        if len(self.portfolio_values) < 2:
            return 0.0

        try:
            # Handle NaN and inf values
            values = self.portfolio_values.replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 2:
                return 0.0

            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            max_dd = abs(drawdown.min())
            return (
                float(max_dd) if not np.isnan(max_dd) and not np.isinf(max_dd) else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_avg_trade_return(self) -> float:
        """Calculate average return per trade."""
        if not self.trades:
            return 0.0

        try:
            returns = []
            for trade in self.trades:
                if trade.initial_value != 0:
                    ret = trade.pnl / trade.initial_value
                    if not np.isnan(ret) and not np.isinf(ret):
                        returns.append(ret)

            if not returns:
                return 0.0

            avg_return = np.mean(returns)
            return (
                float(avg_return)
                if not np.isnan(avg_return) and not np.isinf(avg_return)
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating average trade return: {e}")
            return 0.0

    def calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes."""
        if not self.trades:
            return 0.0

        try:
            durations = [
                trade.holding_time for trade in self.trades if trade.holding_time
            ]
            if not durations:
                return 0.0

            avg_duration = np.mean(durations)
            return (
                float(avg_duration)
                if not np.isnan(avg_duration) and not np.isinf(avg_duration)
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating average trade duration: {e}")
            return 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate ratio of gross profits to gross losses."""
        if not self.trades:
            return 0.0

        try:
            gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
            gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))

            if gross_loss == 0:
                return 100.0 if gross_profit > 0 else 0.0

            profit_factor = gross_profit / gross_loss
            return (
                float(profit_factor)
                if not np.isnan(profit_factor) and not np.isinf(profit_factor)
                else 0.0
            )
        except Exception as e:
            logger.warning(f"Error calculating profit factor: {e}")
            return 0.0

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        try:
            metrics = {
                "sharpe_ratio": self.calculate_sharpe_ratio(),
                "sortino_ratio": self.calculate_sortino_ratio(),
                "win_rate": self.calculate_win_rate(),
                "max_drawdown": self.calculate_max_drawdown(),
                "avg_trade_return": self.calculate_avg_trade_return(),
                "avg_trade_duration": self.calculate_avg_trade_duration(),
                "profit_factor": self.calculate_profit_factor(),
                "total_trades": len(self.trades),
            }

            # Calculate total return with validation
            if len(self.portfolio_values) > 1:
                try:
                    total_return = (
                        self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]
                    ) - 1
                    metrics["total_return"] = (
                        float(total_return)
                        if not np.isnan(total_return) and not np.isinf(total_return)
                        else 0.0
                    )
                except Exception:
                    metrics["total_return"] = 0.0
            else:
                metrics["total_return"] = 0.0

            # Ensure all metrics are valid numbers
            for key in metrics:
                if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                    metrics[key] = 0.0

            return metrics
        except Exception as e:
            logger.warning(f"Error calculating all metrics: {e}")
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "avg_trade_return": 0.0,
                "avg_trade_duration": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "total_return": 0.0,
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
