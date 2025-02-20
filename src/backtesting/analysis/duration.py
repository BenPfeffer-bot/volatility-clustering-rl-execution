"""
Trade duration analysis component that enforces the 10-30 minute holding time target from main.mdc.
Provides monitoring and optimization of trade durations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..strategies.base import Trade
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class TradeDurationAnalyzer:
    """
    Analyzes and optimizes trade durations to meet the 10-30 minute target from main.mdc.
    """

    def __init__(self):
        self.MIN_DURATION = 10  # minutes
        self.MAX_DURATION = 30  # minutes
        self.trades: List[Trade] = []

    def add_trade(self, trade: Trade):
        """Add a completed trade for analysis."""
        self.trades.append(trade)

    def analyze_durations(self) -> Dict[str, float]:
        """
        Analyze trade durations against targets.

        Returns:
            Dict with duration statistics and target achievement metrics
        """
        if not self.trades:
            return {
                "avg_duration": 0.0,
                "pct_within_target": 0.0,
                "pct_too_short": 0.0,
                "pct_too_long": 0.0,
            }

        durations = [trade.holding_time for trade in self.trades if trade.holding_time]

        if not durations:
            return {
                "avg_duration": 0.0,
                "pct_within_target": 0.0,
                "pct_too_short": 0.0,
                "pct_too_long": 0.0,
            }

        durations = np.array(durations)

        return {
            "avg_duration": np.mean(durations),
            "pct_within_target": np.mean(
                (durations >= self.MIN_DURATION) & (durations <= self.MAX_DURATION)
            ),
            "pct_too_short": np.mean(durations < self.MIN_DURATION),
            "pct_too_long": np.mean(durations > self.MAX_DURATION),
        }

    def get_optimization_suggestions(self) -> List[str]:
        """
        Generate suggestions for optimizing trade durations.

        Returns:
            List of optimization suggestions based on duration analysis
        """
        analysis = self.analyze_durations()
        suggestions = []

        # Handle too short durations
        if analysis["pct_too_short"] > 0.2:  # More than 20% too short
            suggestions.extend(
                [
                    "Consider increasing VPIN threshold for entry to ensure more sustained institutional activity",
                    "Adjust profit taking levels to allow for longer holding periods",
                    "Review early exit conditions that may be triggering prematurely",
                ]
            )

        # Handle too long durations
        if analysis["pct_too_long"] > 0.2:  # More than 20% too long
            suggestions.extend(
                [
                    "Consider tightening take-profit levels during high VPIN periods",
                    "Implement more aggressive exit rules when market impact diminishes",
                    "Add additional exit conditions based on volume profile changes",
                ]
            )

        # Specific suggestions based on trade analysis
        if self.trades:
            vpin_analysis = self._analyze_vpin_impact()
            if vpin_analysis["high_vpin_duration"] < self.MIN_DURATION:
                suggestions.append(
                    "High VPIN trades are closing too quickly. Consider adjusting "
                    "exit thresholds for high VPIN conditions."
                )

            if vpin_analysis["low_vpin_duration"] > self.MAX_DURATION:
                suggestions.append(
                    "Low VPIN trades are held too long. Consider more aggressive "
                    "exit rules in low VPIN conditions."
                )

        return suggestions

    def _analyze_vpin_impact(self) -> Dict[str, float]:
        """Analyze how VPIN levels affect trade duration."""
        high_vpin_trades = [
            t.holding_time for t in self.trades if t.holding_time and t.vpin_entry > 0.7
        ]

        low_vpin_trades = [
            t.holding_time
            for t in self.trades
            if t.holding_time and t.vpin_entry <= 0.7
        ]

        return {
            "high_vpin_duration": np.mean(high_vpin_trades) if high_vpin_trades else 0,
            "low_vpin_duration": np.mean(low_vpin_trades) if low_vpin_trades else 0,
        }

    def generate_report(self) -> str:
        """Generate detailed duration analysis report."""
        analysis = self.analyze_durations()
        vpin_impact = self._analyze_vpin_impact()

        report = []
        report.append("Trade Duration Analysis Report")
        report.append("=" * 50)

        # Duration statistics
        report.append("\nDuration Statistics:")
        report.append(f"Average Duration: {analysis['avg_duration']:.1f} minutes")
        report.append(
            f"Within Target Range (10-30 min): {analysis['pct_within_target']:.1%}"
        )
        report.append(f"Too Short (<10 min): {analysis['pct_too_short']:.1%}")
        report.append(f"Too Long (>30 min): {analysis['pct_too_long']:.1%}")

        # VPIN impact
        report.append("\nVPIN Impact Analysis:")
        report.append(
            f"High VPIN (>0.7) Avg Duration: {vpin_impact['high_vpin_duration']:.1f} min"
        )
        report.append(
            f"Low VPIN (≤0.7) Avg Duration: {vpin_impact['low_vpin_duration']:.1f} min"
        )

        # Add optimization suggestions
        report.append("\nOptimization Suggestions:")
        for suggestion in self.get_optimization_suggestions():
            report.append(f"- {suggestion}")

        return "\n".join(report)

    def plot_duration_distribution(self) -> None:
        """Plot distribution of trade durations with target range highlighted."""
        try:
            import matplotlib.pyplot as plt

            durations = [t.holding_time for t in self.trades if t.holding_time]

            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=30, alpha=0.7)
            plt.axvline(
                self.MIN_DURATION, color="r", linestyle="--", label="Min Target"
            )
            plt.axvline(
                self.MAX_DURATION, color="r", linestyle="--", label="Max Target"
            )
            plt.xlabel("Trade Duration (minutes)")
            plt.ylabel("Frequency")
            plt.title("Trade Duration Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
