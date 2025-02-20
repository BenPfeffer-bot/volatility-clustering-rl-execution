"""
Advanced experimentation framework for backtesting analysis.
Implements comprehensive testing scenarios and analysis tools.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Any
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.scripts.test_backtest_system import load_test_data
from src.scripts.test_strategy_comparison import MeanReversionStrategy

from src.backtesting.analysis.drawdown import DrawdownMonitor
from src.backtesting.analysis.duration import TradeDurationAnalyzer
from src.backtesting.analysis.risk import RiskManager
from src.backtesting.analysis.performance import PerformanceAnalyzer
from src.backtesting.analysis.visualization import BacktestVisualizer

from src.backtesting.core.engine import BacktestEngine
from src.backtesting.core.metrics import BacktestMetrics
from src.backtesting.core.portfolio import Portfolio
from src.backtesting.core.trade import Trade

from src.backtesting.execution.impact import MarketImpactModel
from src.backtesting.execution.institutional import InstitutionalExecutionModel

from src.backtesting.optimization.regime import RegimeDetector
from src.backtesting.optimization.bayesian import BayesianOptimizer
from src.backtesting.optimization.position import PositionOptimizer

from src.backtesting.performance.comparison import StrategyComparison
from src.backtesting.performance.metrics import PerformanceMetrics

from src.backtesting.strategies.base import BaseStrategy

from src.backtesting.validation.monte_carlo import MonteCarloSimulator
from src.backtesting.validation.cost_sensitivity import CostSensitivityAnalyzer
from src.backtesting.validation.oos import OutOfSampleTester
from src.backtesting.validation.walk_forward import WalkForwardOptimizer
from src.backtesting.validation.runner import ValidationRunner

from src.config.paths import BACKTESTS_DIR
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class ExperimentManager:
    """
    Enhanced experiment manager that integrates all backtesting components:
    - Analysis (Drawdown, Duration, Risk, Performance, Visualization)
    - Core (Engine, Metrics, Portfolio, Trade)
    - Execution (Impact, Institutional)
    - Optimization (Regime, Bayesian, Position)
    - Performance (Comparison, Metrics)
    - Validation (Monte Carlo, Cost Sensitivity, OOS, Walk Forward)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        output_dir: Optional[Path] = None,
        max_drawdown_limit: float = 0.15,  # 15% maximum drawdown limit
    ):
        self.initial_capital = initial_capital
        self.output_dir = output_dir or BACKTESTS_DIR / "experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_drawdown_limit = max_drawdown_limit
        self.performance_metrics = {}  # Initialize empty performance metrics

        # Initialize components
        self.drawdown_monitor = DrawdownMonitor()
        self.duration_analyzer = TradeDurationAnalyzer()
        self.risk_manager = RiskManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = BacktestVisualizer(self.output_dir)

        # Initialize execution components
        self.impact_model = MarketImpactModel()
        self.execution_model = InstitutionalExecutionModel(
            impact_model=self.impact_model
        )

        # Initialize optimization components
        self.regime_detector = RegimeDetector()
        self.position_optimizer = PositionOptimizer()

        # Initialize validation components
        self.monte_carlo = MonteCarloSimulator()
        self.cost_analyzer = CostSensitivityAnalyzer()
        self.oos_tester = OutOfSampleTester()
        self.walk_forward = WalkForwardOptimizer()
        self.validation_runner = ValidationRunner(None, None, self.output_dir)

        self.experiments_history: Dict[str, Dict] = {}

    def run_comprehensive_analysis(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        benchmark_strategies: Optional[Dict[str, BaseStrategy]] = None,
        parameter_ranges: Optional[Dict] = None,
    ) -> Dict:
        """
        Run comprehensive strategy analysis including all components.

        Args:
            strategy: Main strategy to analyze
            data: Market data for testing
            benchmark_strategies: Optional dict of benchmark strategies
            parameter_ranges: Optional parameter ranges for optimization

        Returns:
            Dict with all analysis results
        """
        logger.info("Starting comprehensive analysis...")
        results = {}

        # 1. Regime Detection and Analysis
        logger.info("Detecting market regimes...")
        vol_regimes = self.regime_detector.detect_regimes(data)
        flow_regimes = self.regime_detector.classify_flow_regime(data)
        trend_regimes = self.regime_detector.detect_trend_regime(data)

        # Combine regimes
        combined_regimes = self.regime_detector.combine_regimes(
            volatility_regime=vol_regimes,
            flow_regime=flow_regimes,
            trend_regime=trend_regimes,
        )

        data["regime"] = combined_regimes

        # 2. Basic Backtest with Enhanced Monitoring
        logger.info("Running basic backtest with enhanced monitoring...")
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission_rate=0.00005,  # 0.005% commission
            slippage_rate=0.00002,  # 0.2bps slippage
            slippage_model="fixed",
        )
        basic_results = engine.run_backtest(data)

        # Store trades and portfolio values in results
        results["trades"] = basic_results["trades"]
        results["portfolio_values"] = basic_results["portfolio_values"]

        # Monitor risk limits
        risk_monitoring = self._monitor_risk_limits(
            strategy, data, basic_results["trades"], basic_results["portfolio_values"]
        )
        results["risk_monitoring"] = risk_monitoring

        # Track drawdowns and durations
        for trade in basic_results["trades"]:
            self.duration_analyzer.add_trade(trade)
            if trade.exit_time:
                self.drawdown_monitor.update(
                    trade.exit_time, basic_results["portfolio_values"][trade.exit_time]
                )

        # 3. Performance Analysis
        logger.info("Analyzing performance...")
        performance_results = self.performance_analyzer.analyze_performance(
            basic_results["trades"], basic_results["portfolio_values"], data
        )
        results["performance"] = performance_results

        # 4. Regime Analysis
        logger.info("Analyzing regime performance...")
        regime_results = self._analyze_market_regimes(
            strategy, data, basic_results["trades"]
        )
        results["regime_analysis"] = regime_results

        # 5. Risk Analysis
        logger.info("Analyzing risk metrics...")
        risk_metrics = self.risk_manager.update_risk_metrics(
            basic_results["portfolio_values"].iloc[-1],
            basic_results["trades"],
            basic_results["portfolio_values"].pct_change().dropna(),
        )
        results["risk"] = risk_metrics

        # 6. Position Optimization
        logger.info("Optimizing positions...")
        position_results = self._optimize_positions(
            strategy, data, basic_results["trades"]
        )
        results["position_optimization"] = position_results

        # 7. Validation Suite
        logger.info("Running validation suite...")
        validation_results = self._run_validation_suite(
            strategy, data, parameter_ranges
        )
        results["validation"] = validation_results

        # 8. Benchmark Comparison
        if benchmark_strategies:
            logger.info("Running benchmark comparison...")
            comparison = StrategyComparison(initial_capital=self.initial_capital)
            benchmark_results = comparison.compare_strategies(
                {"main": strategy, **benchmark_strategies}, data
            )
            results["benchmarks"] = benchmark_results

        # 9. Generate Visualizations
        logger.info("Generating visualizations...")
        report_dir = self.visualizer.generate_performance_report(
            basic_results["portfolio_values"],
            basic_results["trades"],
            self.drawdown_monitor.get_drawdown_statistics()["max_drawdown"],
        )
        results["report_dir"] = str(report_dir)

        # Store results
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiments_history[experiment_id] = results

        return results

    def _optimize_positions(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        trades: List[Trade],
    ) -> Dict:
        """Run position optimization analysis."""
        # Calculate win rate from completed trades
        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0

        # Get current market data
        latest_data = data.iloc[-1]

        # Optimize position size
        optimal_size = self.position_optimizer.calculate_position_size(
            win_rate=win_rate,
            volatility=latest_data["daily_volatility"],
            market_data=latest_data,
            regime=latest_data["regime"],
        )

        # Analyze entry/exit timing
        timing_results = {}
        for trade in trades:
            entry_data = data.loc[trade.entry_time]
            should_enter, timing_metrics = (
                self.position_optimizer.optimize_entry_timing(
                    1, data.loc[: trade.entry_time].tail(20)
                )
            )
            timing_results[trade.entry_time] = {
                "should_enter": should_enter,
                "metrics": timing_metrics,
            }

        return {
            "optimal_size": optimal_size,
            "timing_analysis": timing_results,
            "position_report": self.position_optimizer.generate_position_report(),
        }

    def _run_validation_suite(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        parameter_ranges: Optional[Dict] = None,
    ) -> Dict:
        """Run comprehensive validation suite."""
        # 1. Out-of-sample testing
        oos_results = self.oos_tester.run_oos_test(strategy, data, parameter_ranges)

        # 2. Walk-forward optimization
        if parameter_ranges:
            wfo_results = self.walk_forward.run_wfo(strategy, data, parameter_ranges)
        else:
            wfo_results = None

        # 3. Monte Carlo simulation
        mc_results = self.monte_carlo.run_simulations(strategy.trades)

        # 4. Cost sensitivity analysis
        cost_results = self.cost_analyzer.run_analysis(
            strategy,
            data,
            commission_range=[0.0005, 0.001, 0.002],
            slippage_range=[0.0001, 0.0002, 0.0005],
        )

        return {
            "oos_testing": oos_results,
            "walk_forward": wfo_results,
            "monte_carlo": mc_results,
            "cost_sensitivity": cost_results,
        }

    def _analyze_market_regimes(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        trades: List[Trade],
    ) -> Dict:
        """Analyze strategy performance across different market regimes."""
        results = {}

        # Get flow regimes in addition to volatility regimes
        flow_regimes = self.regime_detector.classify_flow_regime(data)
        trend_regimes = self.regime_detector.detect_trend_regime(data)

        # Combine regimes
        combined_regimes = self.regime_detector.combine_regimes(
            volatility_regime=data["regime"],
            flow_regime=flow_regimes,
            trend_regime=trend_regimes,
        )

        # Analyze each regime type
        regime_types = {
            "volatility": data["regime"],
            "flow": flow_regimes,
            "trend": trend_regimes,
            "combined": combined_regimes,
        }

        for regime_type, regime_series in regime_types.items():
            regime_results = {}
            for regime in regime_series.unique():
                regime_data = data[regime_series == regime]
                regime_trades = [t for t in trades if t.entry_time in regime_data.index]

                if regime_trades:
                    metrics = PerformanceMetrics(regime_trades, regime_data["close"])
                    regime_results[regime] = metrics.calculate_all_metrics()

            results[regime_type] = regime_results

        return results

    def _monitor_risk_limits(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        trades: List[Trade],
        portfolio_values: pd.Series,
    ) -> Dict:
        """Monitor risk limits and generate alerts."""
        alerts = []
        if len(portfolio_values) < 2:
            return {"alerts": alerts, "status": "OK"}

        # Calculate drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = drawdown.max()

        # Check risk limits
        if max_drawdown > self.max_drawdown_limit:
            alerts.append(
                {
                    "type": "RISK_LIMIT_BREACH",
                    "message": f"Max drawdown breached: {max_drawdown:.2%}",
                    "timestamp": pd.Timestamp.now(),
                    "metrics": {
                        "current_drawdown": max_drawdown,
                        "volatility": portfolio_values.pct_change().std()
                        * np.sqrt(252),
                        "sharpe_ratio": self.performance_metrics.get("sharpe_ratio", 0),
                        "sortino_ratio": self.performance_metrics.get(
                            "sortino_ratio", 0
                        ),
                        "win_rate": self.performance_metrics.get("win_rate", 0),
                        "profit_factor": self.performance_metrics.get(
                            "profit_factor", 0
                        ),
                    },
                }
            )

        return {"alerts": alerts, "status": "ALERT" if alerts else "OK"}

    def generate_experiment_report(self, experiment_id: Optional[str] = None) -> str:
        """Generate comprehensive experiment report."""
        if experiment_id is None:
            experiment_id = max(self.experiments_history.keys())

        results = self.experiments_history[experiment_id]

        report = []
        report.append("Enhanced Experiment Report")
        report.append("=" * 50)

        # Performance Metrics
        if "performance" in results and "performance_metrics" in results["performance"]:
            perf = results["performance"]["performance_metrics"]
            report.append("\nPerformance Metrics:")
            if "total_return" in perf:
                report.append(f"Total Return: {perf['total_return']:.2%}")
            if "sharpe_ratio" in perf:
                report.append(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            if "win_rate" in perf:
                report.append(f"Win Rate: {perf['win_rate']:.2%}")

        # Risk Metrics
        if "risk" in results:
            risk = results["risk"]
            report.append("\nRisk Metrics:")
            if "current_drawdown" in risk:
                report.append(f"Current Drawdown: {risk['current_drawdown']:.2%}")
            if "volatility" in risk:
                report.append(f"Volatility: {risk['volatility']:.2%}")
            if "sortino_ratio" in risk:
                report.append(f"Sortino Ratio: {risk['sortino_ratio']:.2f}")

        # Duration Analysis
        duration_stats = self.duration_analyzer.analyze_durations()
        report.append("\nDuration Analysis:")
        if "avg_duration" in duration_stats:
            report.append(
                f"Average Duration: {duration_stats['avg_duration']:.1f} minutes"
            )
        if "pct_within_target" in duration_stats:
            report.append(f"Within Target: {duration_stats['pct_within_target']:.1%}")

        # Validation Results
        if "validation" in results:
            validation = results["validation"]
            report.append("\nValidation Results:")

            # OOS Results
            if "oos_testing" in validation and "analysis" in validation["oos_testing"]:
                oos = validation["oos_testing"]["analysis"]
                report.append("\nOut-of-Sample Testing:")
                if "passes_validation" in oos:
                    report.append(f"Passes Validation: {oos['passes_validation']}")
                if "is_overfit" in oos:
                    report.append(f"Is Overfit: {oos['is_overfit']}")

            # Monte Carlo Results
            if "monte_carlo" in validation and "metrics" in validation["monte_carlo"]:
                mc = validation["monte_carlo"]["metrics"]
                report.append("\nMonte Carlo Simulation:")
                if "worst_case" in mc and "drawdown" in mc["worst_case"]:
                    report.append(
                        f"Worst Drawdown (99th): {mc['worst_case']['drawdown']:.2%}"
                    )
                if "probabilities" in mc and "sharpe_below_1" in mc["probabilities"]:
                    report.append(
                        f"Probability of Loss: {mc['probabilities']['sharpe_below_1']:.2%}"
                    )

            # Cost Sensitivity
            if "cost_sensitivity" in validation:
                cost = validation["cost_sensitivity"]
                report.append("\nCost Sensitivity Analysis:")
                if (
                    "commission_sensitivity" in cost
                    and "impact_analysis" in cost["commission_sensitivity"]
                    and "total_return"
                    in cost["commission_sensitivity"]["impact_analysis"]
                    and "max_impact"
                    in cost["commission_sensitivity"]["impact_analysis"]["total_return"]
                ):
                    report.append(
                        f"Commission Impact: {cost['commission_sensitivity']['impact_analysis']['total_return']['max_impact']:.2%}"
                    )
                if (
                    "slippage_sensitivity" in cost
                    and "impact_analysis" in cost["slippage_sensitivity"]
                    and "total_return"
                    in cost["slippage_sensitivity"]["impact_analysis"]
                    and "max_impact"
                    in cost["slippage_sensitivity"]["impact_analysis"]["total_return"]
                ):
                    report.append(
                        f"Slippage Impact: {cost['slippage_sensitivity']['impact_analysis']['total_return']['max_impact']:.2%}"
                    )

        # Position Optimization
        if "position_optimization" in results:
            pos_opt = results["position_optimization"]
            report.append("\nPosition Optimization:")
            if "optimal_size" in pos_opt:
                report.append(f"Optimal Size: {pos_opt['optimal_size']:.2%}")

        # Benchmark Comparison
        if "benchmarks" in results:
            report.append("\nBenchmark Comparison:")
            for name, metrics in results["benchmarks"].items():
                report.append(f"\n{name}:")
                if "metrics" in metrics:
                    if "sharpe_ratio" in metrics["metrics"]:
                        report.append(
                            f"Sharpe Ratio: {metrics['metrics']['sharpe_ratio']:.2f}"
                        )
                    if "win_rate" in metrics["metrics"]:
                        report.append(f"Win Rate: {metrics['metrics']['win_rate']:.2%}")

        return "\n".join(report)

    def generate_detailed_report(self, experiment_id: Optional[str] = None) -> str:
        """Generate detailed experiment report with all metrics."""
        report = self.generate_experiment_report(experiment_id)
        results = self.experiments_history[
            experiment_id or max(self.experiments_history.keys())
        ]

        # Add regime analysis
        if "regime_analysis" in results:
            report += "\n\nRegime Analysis:"
            for regime_type, regime_results in results["regime_analysis"].items():
                report += f"\n\n{regime_type.title()} Regimes:"
                for regime, metrics in regime_results.items():
                    report += f"\n{regime}:"
                    report += f"\n  Sharpe: {metrics['sharpe_ratio']:.2f}"
                    report += f"\n  Win Rate: {metrics['win_rate']:.2%}"
                    report += f"\n  Avg Return: {metrics['avg_trade_return']:.2%}"

        # Add drawdown analysis
        drawdown_stats = self.drawdown_monitor.get_drawdown_statistics()
        report += "\n\nDrawdown Analysis:"
        report += f"\nMax Drawdown: {drawdown_stats['max_drawdown']:.2%}"
        report += f"\nAvg Drawdown: {drawdown_stats['avg_drawdown']:.2%}"
        report += (
            f"\nAvg Recovery Time: {drawdown_stats['avg_recovery_time']:.1f} minutes"
        )

        # Add duration analysis suggestions
        duration_suggestions = self.duration_analyzer.get_optimization_suggestions()
        if duration_suggestions:
            report += "\n\nDuration Optimization Suggestions:"
            for suggestion in duration_suggestions:
                report += f"\n- {suggestion}"

        # Add risk monitoring status
        if "risk_monitoring" in results:
            risk_mon = results["risk_monitoring"]
            report += "\n\nRisk Monitoring Status:"
            report += f"\nStatus: {risk_mon['status'].upper()}"
            if risk_mon["alerts"]:
                report += "\nActive Alerts:"
                for alert in risk_mon["alerts"]:
                    report += f"\n- {alert['type']}: {alert['message']}"

        return report

    def print_risk_alerts(self, alerts: List[Dict]) -> None:
        """Print risk alerts in a formatted way."""
        if alerts:
            print("\nRisk limits breached!")
            for alert in alerts:
                print(f"Alert: {alert['type']} - {alert['message']}")


if __name__ == "__main__":
    # Example usage
    experiment_mgr = ExperimentManager()
    results = experiment_mgr.run_comprehensive_analysis(
        strategy=MeanReversionStrategy(),
        # data=load_test_data(),
        data=pd.read_csv("data/test_data.csv"),
    )

    # Get regime-specific performance
    regime_analysis = results["regime_analysis"]
    for regime_type, metrics in regime_analysis.items():
        print(f"Performance in {regime_type} regimes:")
        print(metrics)

    # Get risk monitoring status
    risk_status = results["risk_monitoring"]
    if risk_status["alerts"]:
        print("Risk alerts detected:", risk_status["alerts"])

    # Get detailed report
    detailed_report = experiment_mgr.generate_detailed_report()
    print(detailed_report)

    # Monitor risk limits
    risk_monitoring = experiment_mgr._monitor_risk_limits(
        strategy=MeanReversionStrategy(),
        data=load_test_data(),
        trades=results["trades"],
        portfolio_values=results["portfolio_values"],
    )

    if risk_monitoring["alerts"]:
        print("Risk alerts detected:")
        experiment_mgr.print_risk_alerts(risk_monitoring["alerts"])
