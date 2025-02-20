# src/backtesting/performance/comparison.py

"""
Strategy comparison framework for evaluating trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ..strategies.base import Trade, BaseStrategy
from .metrics import PerformanceMetrics
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class StrategyComparison:
    """
    Framework for comparing trading strategies against benchmarks.
    Compares against targets from main.mdc:
    - Enhanced Institutional (Sharpe: 2.85, Return: 0.75%)
    - VWAP Execution (Sharpe: 1.3, Return: 0.3%)
    - Mean Reversion (Sharpe: 1.5, Return: 0.4%)
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital

    def _run_backtest(
        self, strategy: BaseStrategy, data: pd.DataFrame
    ) -> Tuple[List[Trade], pd.Series]:
        """
        Run backtest for a single strategy.

        Args:
            strategy: Strategy to test
            data: Market data for testing

        Returns:
            Tuple of (trades list, portfolio value series)
        """
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        portfolio_values = pd.Series(index=data.index)
        trades: List[Trade] = []
        active_trade: Optional[Trade] = None

        # Generate trading signals
        signals = strategy.generate_signals(data)

        # Simulate trading
        for timestamp, row in data.iterrows():
            # Update portfolio value
            if active_trade:
                # Update trade P&L
                price_change = row["close"] - active_trade.entry_price
                trade_pnl = price_change * active_trade.quantity
                portfolio_value = active_trade.initial_value + trade_pnl

            # Record portfolio value
            portfolio_values[timestamp] = portfolio_value

            # Process signals
            signal = signals[timestamp] if timestamp in signals.index else 0

            if signal != 0 and not active_trade:
                # Enter new trade
                quantity = portfolio_value / row["close"]  # Simple position sizing
                active_trade = Trade(
                    entry_time=timestamp,
                    entry_price=row["close"],
                    quantity=quantity,
                    initial_value=portfolio_value,
                    vpin_entry=row.get("vpin", 0),
                    regime=row.get("regime", "unknown"),
                )

            elif active_trade and (
                signal == 0 or strategy.should_exit(row, active_trade)
            ):
                # Exit trade
                active_trade.exit_time = timestamp
                active_trade.exit_price = row["close"]
                active_trade.pnl = (
                    active_trade.exit_price - active_trade.entry_price
                ) * active_trade.quantity
                active_trade.holding_time = (
                    active_trade.exit_time - active_trade.entry_time
                ).total_seconds() / 60

                trades.append(active_trade)
                active_trade = None

        return trades, portfolio_values

    def compare_strategies(
        self, strategies: Dict[str, BaseStrategy], data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compare multiple strategies.

        Args:
            strategies: Dict of strategy name to strategy object
            data: Market data for testing

        Returns:
            Dict of strategy results
        """
        results = {}

        for name, strategy in strategies.items():
            logger.info(f"Testing strategy: {name}")

            # Run backtest
            trades, portfolio_values = self._run_backtest(strategy, data)

            # Calculate metrics
            metrics = PerformanceMetrics(trades, portfolio_values)
            results[name] = {
                "trades": trades,
                "portfolio_values": portfolio_values,
                "metrics": metrics.calculate_all_metrics(),
                "targets": metrics.check_performance_targets(),
            }

        return results

    def generate_comparison_report(self, results: Dict[str, Dict]) -> str:
        """Generate detailed comparison report."""
        report = []
        report.append("Strategy Comparison Report")
        report.append("=" * 50)

        # Compare key metrics
        metrics_table = []
        metrics_table.append(
            "\nKey Metrics Comparison:\n"
            + "-" * 80
            + "\n"
            + f"{'Strategy':<20} {'Sharpe':<10} {'Win Rate':<10} {'Avg Ret':<10} {'Max DD':<10}"
            + "\n"
            + "-" * 80
        )

        for strategy_name, result in results.items():
            metrics = result["metrics"]
            metrics_table.append(
                f"{strategy_name:<20} "
                f"{metrics['sharpe_ratio']:>9.2f} "
                f"{metrics['win_rate']:>9.1%} "
                f"{metrics['avg_trade_return']:>9.2%} "
                f"{metrics['max_drawdown']:>9.2%}"
            )

        report.extend(metrics_table)

        # Compare target achievement
        report.append("\nTarget Achievement by Strategy:")
        report.append("-" * 50)

        for strategy_name, result in results.items():
            report.append(f"\n{strategy_name}:")
            targets = result["targets"]
            for target, achieved in targets.items():
                report.append(f"✓ {target}: {'Yes' if achieved else 'No'}")

        # Compare to benchmarks
        report.append("\nBenchmark Comparison:")
        report.append("-" * 50)
        benchmarks = {
            "Enhanced Institutional": {"sharpe": 2.85, "return": 0.0075},
            "VWAP Execution": {"sharpe": 1.3, "return": 0.003},
            "Mean Reversion": {"sharpe": 1.5, "return": 0.004},
        }

        for strategy_name, result in results.items():
            metrics = result["metrics"]
            report.append(f"\n{strategy_name} vs Benchmarks:")
            for bench_name, bench_metrics in benchmarks.items():
                sharpe_diff = metrics["sharpe_ratio"] - bench_metrics["sharpe"]
                return_diff = metrics["avg_trade_return"] - bench_metrics["return"]
                report.append(
                    f"vs {bench_name}:"
                    f" Sharpe: {sharpe_diff:+.2f},"
                    f" Return: {return_diff:+.2%}"
                )

        return "\n".join(report)

    def plot_strategy_comparison(
        self, results: Dict[str, Dict], save_path: Optional[str] = None
    ) -> None:
        """Plot strategy comparison charts."""
        try:
            import matplotlib.pyplot as plt

            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Equity curves
            for name, result in results.items():
                portfolio_values = result["portfolio_values"]
                normalized_equity = portfolio_values / portfolio_values.iloc[0]
                ax1.plot(normalized_equity.index, normalized_equity, label=name)
            ax1.set_title("Normalized Equity Curves")
            ax1.legend()
            ax1.grid(True)

            # 2. Drawdown comparison
            for name, result in results.items():
                portfolio_values = result["portfolio_values"]
                peak = portfolio_values.expanding().max()
                drawdown = (portfolio_values - peak) / peak
                ax2.plot(drawdown.index, drawdown, label=name)
            ax2.set_title("Drawdown Comparison")
            ax2.legend()
            ax2.grid(True)

            # 3. Return distribution
            for name, result in results.items():
                returns = [t.pnl / t.initial_value for t in result["trades"]]
                ax3.hist(
                    returns,
                    bins=50,
                    alpha=0.5,
                    label=name,
                    density=True,
                )
            ax3.set_title("Return Distribution")
            ax3.legend()
            ax3.grid(True)

            # 4. Key metrics comparison
            metrics = []
            for name, result in results.items():
                m = result["metrics"]
                metrics.append(
                    [
                        m["sharpe_ratio"],
                        m["win_rate"],
                        m["avg_trade_return"],
                        m["max_drawdown"],
                    ]
                )

            metrics = np.array(metrics)
            x = np.arange(4)
            width = 0.8 / len(results)

            for i, (name, _) in enumerate(results.items()):
                ax4.bar(
                    x + i * width,
                    metrics[i],
                    width,
                    label=name,
                )

            ax4.set_title("Key Metrics Comparison")
            ax4.set_xticks(x + width * (len(results) - 1) / 2)
            ax4.set_xticklabels(
                ["Sharpe", "Win Rate", "Avg Return", "Max DD"], rotation=45
            )
            ax4.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
