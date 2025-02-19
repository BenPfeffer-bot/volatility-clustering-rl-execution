"""
Performance visualization and analysis module.

Key features:
- Equity curve visualization
- Trade analysis plots
- Market impact visualization
- Regime analysis charts
- Risk metrics visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from pathlib import Path

import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config.paths import PLOTS_DIR, BACKTESTS_DIR
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class BacktestVisualizer:
    """Visualization tools for backtest analysis."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.plots_dir = PLOTS_DIR / ticker
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        self.colors = sns.color_palette("husl", 8)

    def plot_equity_curve(self, metrics_history: List[Dict], save: bool = True):
        """Plot equity curve with drawdowns."""
        plt.figure(figsize=(12, 6))

        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Plot equity curve
        plt.plot(
            df.index,
            df["portfolio_value"],
            label="Portfolio Value",
            color=self.colors[0],
        )

        # Add drawdown shading
        running_max = df["portfolio_value"].cummax()
        drawdown = (df["portfolio_value"] - running_max) / running_max
        plt.fill_between(
            df.index,
            df["portfolio_value"],
            running_max,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )

        plt.title(f"Equity Curve - {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()

        if save:
            plt.savefig(self.plots_dir / "equity_curve.png")
            plt.close()
        else:
            plt.show()

    def plot_trade_analysis(self, trades: List[Dict], save: bool = True):
        """Plot trade analysis charts."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. Trade PnL Distribution
        sns.histplot(df["value"], bins=30, ax=ax1, color=self.colors[1])
        ax1.set_title("Trade PnL Distribution")
        ax1.set_xlabel("PnL")

        # 2. Trade Sizes Over Time
        ax2.scatter(
            df["timestamp"],
            df["size"],
            c=df["direction"].map({1: "g", -1: "r"}),
            alpha=0.6,
            label="Trade Size",
        )
        ax2.set_title("Trade Sizes Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Size")

        # 3. Market Impact vs Trade Size
        ax3.scatter(df["size"], df["impact"], alpha=0.6, color=self.colors[2])
        ax3.set_title("Market Impact vs Trade Size")
        ax3.set_xlabel("Trade Size")
        ax3.set_ylabel("Market Impact")

        # 4. Cumulative Returns
        cumulative_pnl = df["value"].cumsum()
        ax4.plot(df["timestamp"], cumulative_pnl, color=self.colors[3])
        ax4.set_title("Cumulative PnL")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Cumulative PnL")

        plt.tight_layout()

        if save:
            plt.savefig(self.plots_dir / "trade_analysis.png")
            plt.close()
        else:
            plt.show()

    def plot_regime_analysis(
        self, metrics_history: List[Dict], regime_changes: List[Dict], save: bool = True
    ):
        """Plot regime analysis and transitions."""
        plt.figure(figsize=(12, 8))

        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Plot returns
        plt.plot(
            df.index, df["returns"], label="Returns", color=self.colors[0], alpha=0.7
        )

        # Add regime shading
        regime_colors = {
            "high_vol": "red",
            "low_vol": "green",
            "institutional": "blue",
            "normal": "gray",
        }

        for regime in regime_changes:
            plt.axvspan(
                regime["start"],
                regime["end"],
                alpha=0.2,
                color=regime_colors[regime["regime"]],
                label=f"{regime['regime']} regime",
            )

        plt.title(f"Regime Analysis - {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()

        if save:
            plt.savefig(self.plots_dir / "regime_analysis.png")
            plt.close()
        else:
            plt.show()

    def plot_risk_metrics(self, risk_metrics: Dict, save: bool = True):
        """Plot risk metrics visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Risk Allocation
        risk_data = {
            "Market Risk": risk_metrics["market_risk"],
            "Impact Risk": risk_metrics["impact_risk"],
            "Liquidity Risk": risk_metrics["liquidity_risk"],
        }

        ax1.pie(
            risk_data.values(),
            labels=risk_data.keys(),
            autopct="%1.1f%%",
            colors=self.colors[:3],
        )
        ax1.set_title("Risk Allocation")

        # 2. Risk Metrics Over Time
        risk_df = pd.DataFrame(risk_metrics["time_series"])
        risk_df.plot(ax=ax2)
        ax2.set_title("Risk Metrics Evolution")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Risk Level")

        plt.tight_layout()

        if save:
            plt.savefig(self.plots_dir / "risk_metrics.png")
            plt.close()
        else:
            plt.show()

    def generate_performance_report(self, backtest_results: Dict, save: bool = True):
        """Generate comprehensive performance report with all visualizations."""
        # Create report directory
        report_dir = (
            BACKTESTS_DIR / self.ticker / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate all plots
        self.plot_equity_curve(backtest_results["metrics_history"])
        self.plot_trade_analysis(backtest_results["trades"])
        self.plot_regime_analysis(
            backtest_results["metrics_history"], backtest_results["regime_changes"]
        )
        self.plot_risk_metrics(backtest_results["risk_report"])

        # Generate performance metrics report
        metrics_report = pd.DataFrame(
            {
                "Metric": [
                    "Total Return",
                    "Sharpe Ratio",
                    "Max Drawdown",
                    "Win Rate",
                    "Total Trades",
                    "Avg Trade Impact",
                    "Total Transaction Costs",
                ],
                "Value": [
                    f"{backtest_results['total_return']:.2%}",
                    f"{backtest_results['sharpe_ratio']:.2f}",
                    f"{backtest_results['max_drawdown']:.2%}",
                    f"{backtest_results['win_rate']:.2%}",
                    backtest_results["total_trades"],
                    f"{backtest_results['avg_trade_impact']:.4f}",
                    f"{backtest_results['total_transaction_costs']:.2f}",
                ],
            }
        )

        # Save metrics report
        metrics_report.to_csv(report_dir / "performance_metrics.csv", index=False)

        # Generate HTML report
        html_report = f"""
        <html>
        <head>
            <title>Backtest Report - {self.ticker}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .metrics {{ margin: 20px 0; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report - {self.ticker}</h1>
            <div class="metrics">
                <h2>Performance Metrics</h2>
                {metrics_report.to_html(index=False)}
            </div>
            <div class="plot">
                <h2>Equity Curve</h2>
                <img src="equity_curve.png" alt="Equity Curve">
            </div>
            <div class="plot">
                <h2>Trade Analysis</h2>
                <img src="trade_analysis.png" alt="Trade Analysis">
            </div>
            <div class="plot">
                <h2>Regime Analysis</h2>
                <img src="regime_analysis.png" alt="Regime Analysis">
            </div>
            <div class="plot">
                <h2>Risk Metrics</h2>
                <img src="risk_metrics.png" alt="Risk Metrics">
            </div>
        </body>
        </html>
        """

        with open(report_dir / "report.html", "w") as f:
            f.write(html_report)

        logger.info(f"Performance report generated at {report_dir}")
        return report_dir
