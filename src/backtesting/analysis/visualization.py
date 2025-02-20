"""
Visualization component for backtest analysis and performance reporting.
Generates comprehensive performance visualizations and reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class BacktestVisualizer:
    """
    Creates visualizations and reports for backtest analysis.
    Focuses on institutional order flow strategy performance metrics.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for all plots
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_equity_curve(
        self,
        portfolio_values: pd.Series,
        drawdowns: pd.Series,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot equity curve with drawdown overlay."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Plot equity curve
        ax1.plot(
            portfolio_values.index, portfolio_values.values, label="Portfolio Value"
        )
        ax1.set_title("Equity Curve")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True, alpha=0.3)

        # Plot drawdown
        ax2.fill_between(
            drawdowns.index,
            0,
            -drawdowns.values * 100,
            color="red",
            alpha=0.3,
            label="Drawdown",
        )
        ax2.set_title("Drawdown")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Drawdown %")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()

    def plot_trade_analysis(
        self, trades: List[Trade], save_path: Optional[Path] = None
    ) -> None:
        """Plot trade analysis charts."""
        if not trades:
            logger.warning("No trades to analyze")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Trade Returns Distribution
        returns = [trade.pnl / trade.initial_value for trade in trades]
        sns.histplot(returns, kde=True, ax=ax1)
        ax1.axvline(0, color="r", linestyle="--")
        ax1.set_title("Trade Returns Distribution")
        ax1.set_xlabel("Return %")

        # 2. Trade Duration vs Return
        durations = [trade.holding_time for trade in trades if trade.holding_time]
        ax2.scatter(durations, returns)
        ax2.axhline(0, color="r", linestyle="--")
        ax2.axvline(10, color="g", linestyle="--", label="Min Target")
        ax2.axvline(30, color="g", linestyle="--", label="Max Target")
        ax2.set_title("Trade Duration vs Return")
        ax2.set_xlabel("Duration (minutes)")
        ax2.set_ylabel("Return %")
        ax2.legend()

        # 3. VPIN vs Return
        vpins = [trade.vpin_entry for trade in trades]
        ax3.scatter(vpins, returns)
        ax3.axhline(0, color="r", linestyle="--")
        ax3.axvline(0.7, color="g", linestyle="--", label="VPIN Threshold")
        ax3.set_title("VPIN vs Return")
        ax3.set_xlabel("VPIN")
        ax3.set_ylabel("Return %")
        ax3.legend()

        # 4. Cumulative Returns
        cumulative_returns = np.cumsum(returns)
        ax4.plot(cumulative_returns)
        ax4.set_title("Cumulative Returns")
        ax4.set_xlabel("Trade Number")
        ax4.set_ylabel("Cumulative Return %")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()

    def plot_regime_analysis(
        self, trades: List[Trade], save_path: Optional[Path] = None
    ) -> None:
        """Plot regime-specific performance analysis."""
        if not trades:
            logger.warning("No trades to analyze")
            return

        # Group trades by regime
        regime_trades = {}
        for trade in trades:
            if trade.regime not in regime_trades:
                regime_trades[trade.regime] = []
            regime_trades[trade.regime].append(trade)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Regime Returns Boxplot
        regime_returns = {
            regime: [t.pnl / t.initial_value for t in trades]
            for regime, trades in regime_trades.items()
        }
        ax1.boxplot(regime_returns.values(), labels=regime_returns.keys())
        ax1.set_title("Returns by Regime")
        ax1.set_ylabel("Return %")
        ax1.grid(True, alpha=0.3)

        # 2. Regime Duration Boxplot
        regime_durations = {
            regime: [t.holding_time for t in trades if t.holding_time]
            for regime, trades in regime_trades.items()
        }
        ax2.boxplot(regime_durations.values(), labels=regime_durations.keys())
        ax2.axhline(10, color="g", linestyle="--", label="Min Target")
        ax2.axhline(30, color="g", linestyle="--", label="Max Target")
        ax2.set_title("Duration by Regime")
        ax2.set_ylabel("Duration (minutes)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()

    def generate_performance_report(
        self,
        portfolio_values: pd.Series,
        trades: List[Trade],
        drawdowns: pd.Series,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Generate comprehensive performance report with visualizations.

        Args:
            portfolio_values: Series of portfolio values
            trades: List of completed trades
            drawdowns: Series of drawdown values
            output_dir: Optional directory for report output

        Returns:
            Path to generated report directory
        """
        # Create report directory
        report_dir = (
            output_dir
            or self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate plots
        self.plot_equity_curve(
            portfolio_values, drawdowns, report_dir / "equity_curve.png"
        )

        self.plot_trade_analysis(trades, report_dir / "trade_analysis.png")

        self.plot_regime_analysis(trades, report_dir / "regime_analysis.png")

        # Generate summary statistics
        stats = self._calculate_summary_statistics(portfolio_values, trades)

        # Save summary report
        self._save_summary_report(stats, report_dir / "summary.txt")

        logger.info(f"Performance report generated at {report_dir}")
        return report_dir

    def _calculate_summary_statistics(
        self, portfolio_values: pd.Series, trades: List[Trade]
    ) -> Dict:
        """Calculate summary statistics for report."""
        returns = portfolio_values.pct_change().dropna()

        stats = {
            "total_return": (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
            "sharpe_ratio": np.sqrt(252) * returns.mean() / returns.std(),
            "num_trades": len(trades),
            "win_rate": sum(1 for t in trades if t.pnl > 0) / len(trades),
            "avg_trade_return": np.mean([t.pnl / t.initial_value for t in trades]),
            "max_drawdown": abs(min(returns.cumsum().min(), 0)),
            "avg_holding_time": np.mean(
                [t.holding_time for t in trades if t.holding_time]
            ),
        }

        # Add target achievement flags
        stats.update(
            {
                "meets_sharpe_target": stats["sharpe_ratio"] > 2.5,
                "meets_win_rate_target": stats["win_rate"] > 0.68,
                "meets_return_target": stats["avg_trade_return"] > 0.0075,
                "meets_duration_target": 10 <= stats["avg_holding_time"] <= 30,
            }
        )

        return stats

    def _save_summary_report(self, stats: Dict, file_path: Path) -> None:
        """Save summary statistics to file."""
        with open(file_path, "w") as f:
            f.write("Performance Summary Report\n")
            f.write("=" * 50 + "\n\n")

            # Core metrics
            f.write("Core Performance Metrics:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Return: {stats['total_return']:.2%}\n")
            f.write(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n")
            f.write(f"Number of Trades: {stats['num_trades']}\n")
            f.write(f"Win Rate: {stats['win_rate']:.2%}\n")
            f.write(f"Average Trade Return: {stats['avg_trade_return']:.2%}\n")
            f.write(f"Maximum Drawdown: {stats['max_drawdown']:.2%}\n")
            f.write(
                f"Average Holding Time: {stats['avg_holding_time']:.1f} minutes\n\n"
            )

            # Target achievement
            f.write("Target Achievement:\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"✓ Sharpe Ratio > 2.5: {'Yes' if stats['meets_sharpe_target'] else 'No'}\n"
            )
            f.write(
                f"✓ Win Rate > 68%: {'Yes' if stats['meets_win_rate_target'] else 'No'}\n"
            )
            f.write(
                f"✓ Avg Return > 0.75%: {'Yes' if stats['meets_return_target'] else 'No'}\n"
            )
            f.write(
                f"✓ Duration 10-30 min: {'Yes' if stats['meets_duration_target'] else 'No'}\n"
            )
