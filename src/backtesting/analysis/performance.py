"""
Performance analysis module for backtesting results.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from ..core.metrics import BacktestMetrics


class PerformanceAnalyzer:
    """Analyzes trading performance and generates reports."""

    def __init__(self):
        """Initialize the performance analyzer."""
        pass

    def analyze_performance(self, trades, portfolio_values, market_data):
        """
        Analyze trading performance and generate a comprehensive report.

        Args:
            trades (List[Trade]): List of completed trades
            portfolio_values (pd.Series): Portfolio values over time
            market_data (pd.DataFrame): Market data used in trading

        Returns:
            Dict[str, Any]: Performance analysis results
        """
        if not trades:
            return {
                "performance_metrics": {},
                "risk_metrics": {},
                "trade_analysis": {},
                "market_impact_analysis": {},
            }

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(portfolio_values)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_values)

        # Analyze trades
        trade_analysis = self._analyze_trades(trades)

        # Analyze market impact
        market_impact_analysis = self._analyze_market_impact(trades)

        return {
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "trade_analysis": trade_analysis,
            "market_impact_analysis": market_impact_analysis,
        }

    def _calculate_performance_metrics(self, portfolio_values):
        """
        Calculate key performance metrics.

        Args:
            portfolio_values (pd.Series): Portfolio values over time

        Returns:
            Dict[str, float]: Performance metrics
        """
        returns = portfolio_values.pct_change().dropna()

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe_ratio = (
            np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        )

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
        }

    def _calculate_risk_metrics(self, portfolio_values):
        """
        Calculate risk metrics.

        Args:
            portfolio_values (pd.Series): Portfolio values over time

        Returns:
            Dict[str, float]: Risk metrics
        """
        returns = portfolio_values.pct_change().dropna()

        # Calculate drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)

        # Calculate value at risk (VaR)
        var_95 = np.percentile(returns, 5)

        return {
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "var_95": var_95,
        }

    def _analyze_trades(self, trades):
        """
        Analyze trading activity.

        Args:
            trades (List[Trade]): List of completed trades

        Returns:
            Dict[str, Any]: Trade analysis results
        """
        if not trades:
            return {}

        # Calculate trade metrics
        n_trades = len(trades)
        profitable_trades = sum(1 for trade in trades if trade.pnl > 0)
        win_rate = profitable_trades / n_trades if n_trades > 0 else 0

        # Calculate average trade duration
        durations = []
        for trade in trades:
            duration = (
                trade.exit_time - trade.entry_time
            ).total_seconds() / 60  # Convert to minutes
            durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # Calculate profit metrics
        pnls = [trade.pnl for trade in trades]
        avg_profit = np.mean(pnls) if pnls else 0
        max_profit = max(pnls) if pnls else 0
        min_profit = min(pnls) if pnls else 0

        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_duration": avg_duration,
            "avg_profit": avg_profit,
            "max_profit": max_profit,
            "min_profit": min_profit,
        }

    def _analyze_market_impact(self, trades):
        """
        Analyze market impact of trades.

        Args:
            trades (List[Trade]): List of completed trades

        Returns:
            Dict[str, float]: Market impact analysis
        """
        if not trades:
            return {}

        # Calculate average market impact
        temp_impacts = [trade.temp_impact for trade in trades]
        perm_impacts = [trade.perm_impact for trade in trades]

        avg_temp_impact = np.mean(temp_impacts) if temp_impacts else 0
        avg_perm_impact = np.mean(perm_impacts) if perm_impacts else 0

        return {"avg_temp_impact": avg_temp_impact, "avg_perm_impact": avg_perm_impact}

    def generate_performance_report(self, analysis_results):
        """
        Generate a formatted performance report.

        Args:
            analysis_results (Dict[str, Any]): Results from analyze_performance

        Returns:
            str: Formatted performance report
        """
        report = []

        # Performance metrics section
        perf_metrics = analysis_results.get("performance_metrics", {})
        report.append("Performance Metrics:")
        report.append(f"Total Return: {perf_metrics.get('total_return', 0):.2%}")
        report.append(
            f"Annualized Return: {perf_metrics.get('annualized_return', 0):.2%}"
        )
        report.append(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio', 0):.2f}")
        report.append("")

        # Risk metrics section
        risk_metrics = analysis_results.get("risk_metrics", {})
        report.append("Risk Metrics:")
        report.append(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Volatility: {risk_metrics.get('volatility', 0):.2%}")
        report.append(f"95% VaR: {risk_metrics.get('var_95', 0):.2%}")
        report.append("")

        # Trade analysis section
        trade_analysis = analysis_results.get("trade_analysis", {})
        report.append("Trade Analysis:")
        report.append(f"Number of Trades: {trade_analysis.get('n_trades', 0)}")
        report.append(f"Win Rate: {trade_analysis.get('win_rate', 0):.2%}")
        report.append(
            f"Average Duration (min): {trade_analysis.get('avg_duration', 0):.1f}"
        )
        report.append(f"Average Profit: {trade_analysis.get('avg_profit', 0):.2f}")
        report.append("")

        # Market impact section
        impact_analysis = analysis_results.get("market_impact_analysis", {})
        report.append("Market Impact Analysis:")
        report.append(
            f"Average Temporary Impact: {impact_analysis.get('avg_temp_impact', 0):.4%}"
        )
        report.append(
            f"Average Permanent Impact: {impact_analysis.get('avg_perm_impact', 0):.4%}"
        )

        return "\n".join(report)
