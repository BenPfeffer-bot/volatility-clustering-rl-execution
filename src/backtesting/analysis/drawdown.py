"""
Drawdown monitoring system that enforces the 4% max drawdown target from main.mdc.
Provides real-time drawdown tracking and risk alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DrawdownEvent:
    """Represents a significant drawdown event."""

    start_time: datetime
    end_time: Optional[datetime]
    max_drawdown: float
    duration: int  # in minutes
    recovery_time: Optional[int] = None  # in minutes
    triggered_stop: bool = False


class DrawdownMonitor:
    """
    Real-time drawdown monitoring system.
    Enforces 4% max drawdown target from main.mdc.
    """

    def __init__(self, max_drawdown_threshold: float = 0.04):
        self.max_drawdown_threshold = max_drawdown_threshold
        self.current_peak = 0.0
        self.current_drawdown = 0.0
        self.drawdown_start: Optional[datetime] = None
        self.drawdown_events: List[DrawdownEvent] = []
        self.in_drawdown = False

    def update(self, timestamp: datetime, portfolio_value: float) -> Optional[Dict]:
        """
        Update drawdown metrics with new portfolio value.

        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value

        Returns:
            Dict with alert if drawdown threshold is breached
        """
        # Update peak if new high
        if portfolio_value > self.current_peak:
            self.current_peak = portfolio_value

            # Check if this ends a drawdown period
            if self.in_drawdown:
                self._record_drawdown_recovery(timestamp)
                self.in_drawdown = False

        # Calculate current drawdown
        if self.current_peak > 0:
            self.current_drawdown = (
                self.current_peak - portfolio_value
            ) / self.current_peak

            # Check for drawdown start
            if not self.in_drawdown and self.current_drawdown > 0:
                self.in_drawdown = True
                self.drawdown_start = timestamp

            # Check for threshold breach
            if self.current_drawdown >= self.max_drawdown_threshold:
                return self._generate_drawdown_alert()

        return None

    def _record_drawdown_recovery(self, recovery_time: datetime):
        """Record drawdown recovery details."""
        if self.drawdown_start:
            duration = (recovery_time - self.drawdown_start).total_seconds() / 60

            event = DrawdownEvent(
                start_time=self.drawdown_start,
                end_time=recovery_time,
                max_drawdown=self.current_drawdown,
                duration=int(duration),
                recovery_time=int(duration),
                triggered_stop=self.current_drawdown >= self.max_drawdown_threshold,
            )

            self.drawdown_events.append(event)
            self.drawdown_start = None

    def _generate_drawdown_alert(self) -> Dict:
        """Generate alert when drawdown threshold is breached."""
        return {
            "type": "DRAWDOWN_ALERT",
            "current_drawdown": self.current_drawdown,
            "threshold": self.max_drawdown_threshold,
            "start_time": self.drawdown_start,
            "duration_minutes": (
                int((datetime.now() - self.drawdown_start).total_seconds() / 60)
                if self.drawdown_start
                else 0
            ),
        }

    def get_drawdown_statistics(self) -> Dict[str, float]:
        """Calculate drawdown statistics."""
        if not self.drawdown_events:
            return {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "avg_duration": 0.0,
                "avg_recovery_time": 0.0,
                "stop_trigger_rate": 0.0,
            }

        max_dd = max(event.max_drawdown for event in self.drawdown_events)
        avg_dd = np.mean([event.max_drawdown for event in self.drawdown_events])
        avg_duration = np.mean([event.duration for event in self.drawdown_events])
        recovery_times = [
            event.recovery_time
            for event in self.drawdown_events
            if event.recovery_time is not None
        ]
        avg_recovery = np.mean(recovery_times) if recovery_times else 0.0
        stop_rate = np.mean(
            [1.0 if event.triggered_stop else 0.0 for event in self.drawdown_events]
        )

        return {
            "max_drawdown": max_dd,
            "avg_drawdown": avg_dd,
            "avg_duration": avg_duration,
            "avg_recovery_time": avg_recovery,
            "stop_trigger_rate": stop_rate,
        }

    def generate_report(self) -> str:
        """Generate detailed drawdown analysis report."""
        stats = self.get_drawdown_statistics()

        report = []
        report.append("Drawdown Analysis Report")
        report.append("=" * 50)

        # Add summary statistics
        report.append("\nSummary Statistics:")
        report.append(f"Maximum Drawdown: {stats['max_drawdown']:.2%}")
        report.append(f"Average Drawdown: {stats['avg_drawdown']:.2%}")
        report.append(f"Average Duration: {stats['avg_duration']:.1f} minutes")
        report.append(
            f"Average Recovery Time: {stats['avg_recovery_time']:.1f} minutes"
        )
        report.append(f"Stop-Loss Trigger Rate: {stats['stop_trigger_rate']:.2%}")

        # Add threshold analysis
        report.append("\nThreshold Analysis:")
        report.append(f"Target Max Drawdown: {self.max_drawdown_threshold:.2%}")
        threshold_breaches = sum(1 for e in self.drawdown_events if e.triggered_stop)
        report.append(f"Threshold Breaches: {threshold_breaches}")

        # Add recent events
        report.append("\nRecent Drawdown Events:")
        for event in sorted(self.drawdown_events[-5:], key=lambda x: x.start_time):
            report.append(
                f"- {event.start_time.strftime('%Y-%m-%d %H:%M')}: "
                f"{event.max_drawdown:.2%} ({event.duration} mins)"
            )

        return "\n".join(report)
