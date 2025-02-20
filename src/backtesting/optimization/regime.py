"""
Market regime detection module for classifying market states.
Implements HMM-based regime detection and VPIN-based flow classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class RegimeDetector:
    """
    Market regime detection system that identifies:
    - Institutional flow regimes
    - Volatility regimes
    - Trend regimes
    - Combined market states
    """

    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 100,
        vpin_threshold: float = 0.7,
        vol_threshold: float = 0.15,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.vpin_threshold = vpin_threshold
        self.vol_threshold = vol_threshold
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.regime_history: Dict[str, pd.Series] = {}

    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Prepare features for regime detection.

        Args:
            data: Market data with features

        Returns:
            Tuple of (scaled_features, feature_index)
        """
        # Calculate features
        features = pd.DataFrame(index=data.index)

        # Volatility features
        features["volatility"] = data["daily_volatility"]
        features["vpin"] = data["vpin"]
        features["returns"] = data["close"].pct_change()

        # Add lagged features
        for col in ["volatility", "vpin", "returns"]:
            features[f"{col}_lag1"] = features[col].shift(1)
            features[f"{col}_lag2"] = features[col].shift(2)

        # Drop NaN values from feature calculation
        features = features.dropna()
        feature_index = features.index

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        return scaled_features, feature_index

    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect market regimes using HMM.

        Args:
            data: Market data with features

        Returns:
            Series of regime labels
        """
        # Prepare features
        features, feature_index = self._prepare_features(data)

        if len(features) == 0:
            return pd.Series(index=data.index)

        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
        )

        # Fit model and predict states
        model.fit(features)
        states = model.predict(features)

        # Map states to regime labels
        state_volatilities = []
        for state in range(self.n_regimes):
            state_mask = states == state
            if np.any(state_mask):
                state_vol = np.mean(
                    features[state_mask, 0]
                )  # Use first feature (volatility)
                state_volatilities.append((state, state_vol))

        # Sort states by volatility
        sorted_states = sorted(state_volatilities, key=lambda x: x[1])
        state_map = {orig_state: i for i, (orig_state, _) in enumerate(sorted_states)}

        # Create regime series with proper index
        regimes = pd.Series("unknown", index=data.index)
        regime_labels = ["low_vol", "neutral", "high_vol", "extreme"]
        regimes.loc[feature_index] = [regime_labels[state_map[s]] for s in states]

        return regimes

    def classify_flow_regime(
        self,
        data: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Classify institutional flow regimes.

        Args:
            data: Market data with VPIN
            window: Rolling window for smoothing

        Returns:
            Series of flow regime labels
        """
        if "vpin" not in data.columns:
            return pd.Series(index=data.index, data="unknown")

        # Smooth VPIN
        vpin_ma = data["vpin"].rolling(window).mean()

        # Classify regimes
        conditions = [
            (vpin_ma > self.vpin_threshold)
            & (data["daily_volatility"] < self.vol_threshold),
            (vpin_ma > self.vpin_threshold)
            & (data["daily_volatility"] >= self.vol_threshold),
            (vpin_ma <= self.vpin_threshold)
            & (data["daily_volatility"] < self.vol_threshold),
            (vpin_ma <= self.vpin_threshold)
            & (data["daily_volatility"] >= self.vol_threshold),
        ]
        choices = [
            "institutional_flow",
            "high_impact",
            "low_impact",
            "high_volatility",
        ]

        # Initialize with default value and ensure consistent dtype
        flow_regimes = pd.Series(
            np.select(conditions, choices, default="unknown"),
            index=data.index,
            dtype=str,
        )

        return flow_regimes

    def detect_trend_regime(
        self,
        data: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Detect trend regimes.

        Args:
            data: Market data
            window: Rolling window for trend calculation

        Returns:
            Series of trend regime labels
        """
        # Calculate trend indicators
        returns = data["close"].pct_change()
        ma_fast = data["close"].rolling(window).mean()
        ma_slow = data["close"].rolling(window * 2).mean()

        # Trend strength
        trend_strength = returns.rolling(window).mean().abs().rolling(window).mean()

        # Classify trends
        conditions = [
            (ma_fast > ma_slow) & (trend_strength > 0.001),
            (ma_fast < ma_slow) & (trend_strength > 0.001),
            (trend_strength <= 0.001),
        ]
        choices = np.array(["uptrend", "downtrend", "ranging"], dtype=str)

        trend_regimes = pd.Series(
            np.select(conditions, choices, default="neutral"),
            index=data.index,
            dtype=str,
        )
        return trend_regimes

    def combine_regimes(
        self,
        volatility_regime: pd.Series,
        flow_regime: pd.Series,
        trend_regime: pd.Series,
    ) -> pd.Series:
        """
        Combine different regime indicators.

        Args:
            volatility_regime: Volatility regime labels
            flow_regime: Flow regime labels
            trend_regime: Trend regime labels

        Returns:
            Combined regime labels
        """
        combined = pd.Series(index=volatility_regime.index, data="unknown")

        # Priority rules for regime combination
        for idx in combined.index:
            vol = volatility_regime[idx]
            flow = flow_regime[idx]
            trend = trend_regime[idx]

            if flow == "institutional_flow":
                combined[idx] = "institutional_flow"
            elif vol == "extreme_volatility":
                combined[idx] = "high_impact"
            elif flow == "high_impact":
                combined[idx] = "high_impact"
            elif trend in ["uptrend", "downtrend"]:
                combined[idx] = "trending"
            else:
                combined[idx] = "neutral"

        return combined

    def update_regime_history(
        self,
        timestamp: pd.Timestamp,
        regimes: Dict[str, str],
    ) -> None:
        """Update regime history."""
        self.regime_history[timestamp.strftime("%Y%m%d_%H%M%S")] = pd.Series(regimes)

    def generate_regime_report(self) -> str:
        """Generate regime analysis report."""
        if not self.regime_history:
            return "No regime history available"

        # Convert history to DataFrame
        history_df = pd.DataFrame(self.regime_history).T

        report = []
        report.append("Market Regime Analysis Report")
        report.append("=" * 50)

        # Regime distribution
        report.append("\nRegime Distribution:")
        for column in history_df.columns:
            dist = history_df[column].value_counts(normalize=True)
            report.append(f"\n{column} Regimes:")
            for regime, freq in dist.items():
                report.append(f"  {regime}: {freq:.1%}")

        # Regime transitions
        report.append("\nRegime Transitions:")
        for column in history_df.columns:
            transitions = history_df[column].value_counts()
            report.append(f"\n{column} Transitions:")
            for regime, count in transitions.items():
                report.append(f"  {regime}: {count}")

        return "\n".join(report)
