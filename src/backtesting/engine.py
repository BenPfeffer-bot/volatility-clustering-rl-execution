"""
Event-driven backtesting engine with market impact simulation.

Key features:
- Event-driven architecture
- Market impact modeling
- Transaction cost analysis
- Performance evaluation
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.execution.position_sizer import PositionConfig
from src.utils.log_utils import setup_logging
from .visualizer import BacktestVisualizer
from src.execution.optimizer import ExecutionOptimizer, ExecutionConfig
from src.models.tcn_impact import MarketImpactPredictor
from src.data.process_features import FeatureEngineering

logger = setup_logging(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    ticker: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    transaction_cost: float = 0.001  # 10 bps
    market_impact_decay: float = 0.5  # Impact decay factor
    min_trade_spacing: int = 5  # Minimum minutes between trades


class MarketSimulator:
    """
    Simulates market behavior with impact modeling.

    Features:
    1. Price impact simulation
    2. Volume profile modeling
    3. Transaction cost analysis
    4. Liquidity constraints
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_price: float = 0.0
        self.current_volume: float = 0.0
        self.impact_history: List[Dict] = []

    def calculate_market_impact(
        self, price: float, volume: float, trade_size: float, direction: int
    ) -> Tuple[float, float]:
        """
        Calculate price impact of a trade.

        Args:
            price: Current price
            volume: Current volume
            trade_size: Size of the trade
            direction: Trade direction (1: buy, -1: sell)

        Returns:
            Tuple of (execution_price, market_impact)
        """
        # Calculate temporary impact
        volume_ratio = trade_size / volume
        base_impact = 0.1 * np.sqrt(volume_ratio)  # Square root impact model

        # Adjust for recent impacts
        decay_factor = self._calculate_decay_factor()
        total_impact = base_impact * (1 + decay_factor)

        # Calculate execution price
        impact = total_impact * direction
        execution_price = price * (1 + impact)

        # Record impact
        self.impact_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "base_impact": base_impact,
                "total_impact": total_impact,
                "volume_ratio": volume_ratio,
            }
        )

        return execution_price, impact

    def _calculate_decay_factor(self) -> float:
        """Calculate impact decay based on recent trading history."""
        if not self.impact_history:
            return 0.0

        recent_impacts = [
            impact["total_impact"]
            * np.exp(
                -self.config.market_impact_decay
                * (pd.Timestamp.now() - impact["timestamp"]).total_seconds()
                / 60
            )
            for impact in self.impact_history[-5:]  # Look at last 5 trades
        ]

        return sum(recent_impacts)


class Backtester:
    """
    Event-driven backtesting engine.

    Features:
    1. Market simulation
    2. Strategy execution
    3. Performance analysis
    4. Transaction cost modeling
    """

    def __init__(
        self,
        config: BacktestConfig,
        execution_config: ExecutionConfig,
        market_impact_model: MarketImpactPredictor,
    ):
        self.config = config
        self.market_simulator = MarketSimulator(config)
        self.execution_optimizer = ExecutionOptimizer(execution_config)
        self.market_impact_model = market_impact_model

        # Performance tracking
        self.portfolio_value: float = config.initial_capital
        self.cash: float = config.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.metrics_history: List[Dict] = []
        self.regime_changes: List[Dict] = []

        # Initialize visualizer

        self.visualizer = BacktestVisualizer(config.ticker)

    def run(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV and feature data

        Returns:
            Dict with backtest results and performance report
        """
        logger.info("Starting backtest...")

        last_regime = None
        regime_start = None

        for idx, row in data.iterrows():
            # Update market state
            self._update_market_state(row)

            # Generate features
            features = self._extract_features(row)

            # Predict market impact
            market_impact_pred = self.market_impact_model.predict(
                pd.DataFrame([features])
            )[0]

            # Get trading decision
            decision = self.execution_optimizer.process_market_update(
                pd.Timestamp(idx), row["close"], features, market_impact_pred
            )

            # Track regime changes
            current_regime = self.execution_optimizer.current_regime
            if current_regime != last_regime:
                if last_regime is not None:
                    self.regime_changes.append(
                        {
                            "regime": last_regime,
                            "start": regime_start,
                            "end": pd.Timestamp(idx),
                        }
                    )
                last_regime = current_regime
                regime_start = pd.Timestamp(idx)

            # Execute decision
            if decision:
                self._execute_trade(decision, row)

            # Update metrics
            self._update_metrics(row)

        # Add final regime period
        if last_regime is not None:
            self.regime_changes.append(
                {"regime": last_regime, "start": regime_start, "end": pd.Timestamp(idx)}
            )

        # Generate final results
        results = self._generate_results()

        # Generate performance report
        report_dir = self.visualizer.generate_performance_report(results)
        results["report_dir"] = report_dir

        logger.info(f"Backtest completed. Performance report available at {report_dir}")
        return results

    def _update_market_state(self, row: pd.Series):
        """Update current market state."""
        self.market_simulator.current_price = row["close"]
        self.market_simulator.current_volume = row["volume"]

    def _extract_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract features from market data."""
        return {
            "log_returns": row["log_returns"],
            "daily_volatility": row["daily_volatility"],
            "vpin": row["vpin"],
            "vwpd": row["vwpd"],
            "asc": row["asc"],
            "hurst": row["hurst"],
        }

    def _execute_trade(self, decision: Dict, row: pd.Series):
        """Execute trading decision with market impact."""
        # Calculate execution price with impact
        execution_price, impact = self.market_simulator.calculate_market_impact(
            row["close"], row["volume"], decision["size"], decision["direction"]
        )

        # Calculate transaction cost
        transaction_cost = (
            execution_price * decision["size"] * self.config.transaction_cost
        )

        # Update portfolio
        trade_value = execution_price * decision["size"]
        self.cash -= trade_value + transaction_cost

        # Record trade
        trade = {
            "timestamp": decision["timestamp"],
            "type": decision["type"],
            "direction": decision["direction"],
            "size": decision["size"],
            "price": execution_price,
            "impact": impact,
            "transaction_cost": transaction_cost,
            "value": trade_value,
        }
        self.trades.append(trade)

    def _update_metrics(self, row: pd.Series):
        """Update performance metrics."""
        # Calculate portfolio value
        position_value = sum(
            pos["size"] * row["close"] for pos in self.positions.values()
        )
        self.portfolio_value = self.cash + position_value

        # Record metrics
        metrics = {
            "timestamp": row.name,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_value": position_value,
            "returns": (self.portfolio_value - self.config.initial_capital)
            / self.config.initial_capital,
        }
        self.metrics_history.append(metrics)

    def _generate_results(self) -> Dict:
        """Generate final backtest results."""
        # Get performance metrics
        performance = self.execution_optimizer.get_performance_metrics()
        risk_report = self.execution_optimizer.get_risk_report()

        # Calculate additional metrics
        returns = pd.Series([m["returns"] for m in self.metrics_history])

        results = {
            "performance": performance,
            "risk_report": risk_report,
            "total_return": returns.iloc[-1],
            "sharpe_ratio": performance.get("sharpe_ratio", 0),
            "max_drawdown": performance.get("max_drawdown", 0),
            "win_rate": performance.get("win_rate", 0),
            "total_trades": len(self.trades),
            "avg_trade_impact": np.mean([t["impact"] for t in self.trades]),
            "total_transaction_costs": sum(t["transaction_cost"] for t in self.trades),
            "metrics_history": self.metrics_history,
            "trades": self.trades,
            "regime_changes": self.regime_changes,
        }

        return results
