"""
Refining and extending our institutional order flow impact trading strategy by:

Improving Feature Engineering: Detect stealthy institutional trades more precisely.
Enhancing the ML Model: Optimize Temporal Convolutional Networks (TCN).
Optimizing Trade Execution: Use Bayesian Optimization for adaptive sizing.
Implementing Risk Management & Regime Adaptation: Avoid noisy environments.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from tqdm import tqdm  # Add progress bar support

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config.paths import RAW_DIR, PROCESSED_DIR
from src.config.settings import DJ_TITANS_50_TICKER
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class FeatureEngineering:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.output_dir = PROCESSED_DIR
        self.df = None

    def load_data(self, ticker: str) -> pd.DataFrame:
        """
        Load and preprocess 1-minute market data for a given ticker.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            pd.DataFrame: Preprocessed dataframe with log returns and volatility

        Raises:
            ValueError: If required columns are missing from the data
        """
        # Load raw data
        df = pd.read_csv(RAW_DIR / f"{ticker}_1min.csv")

        # Validate required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in {ticker}_1min.csv")

        # Calculate log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["log_returns"] = df["log_returns"].replace([np.inf, -np.inf], np.nan)
        df["log_returns"] = df["log_returns"].fillna(0)

        # Calculate daily volatility
        df["daily_volatility"] = df["close"].rolling(window=21).std()
        df["daily_volatility"] = df["daily_volatility"].replace(
            [np.inf, -np.inf], np.nan
        )
        df["daily_volatility"] = df["daily_volatility"].fillna(0)
        logger.info(f"Loaded raw data for {self.ticker}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns}")
        return df

    def calculate_vpin(self, volume_bucket_size: int = 50) -> pd.Series:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN measures order flow toxicity by calculating buy/sell volume imbalance
        over volume-based time buckets.

        Args:
            volume_bucket_size (int): Number of volume buckets to use

        Returns:
            pd.Series: VPIN values
        """
        logger.info("Calculating VPIN...")
        df = self.df.copy()

        # Vectorized operations instead of loops
        price_change = df["close"].diff()

        # Vectorized buy/sell volume calculation
        df["buy_volume"] = df["volume"].where(
            price_change > 0, df["volume"].where(price_change == 0, 0) * 0.5
        )
        df["sell_volume"] = df["volume"].where(
            price_change < 0, df["volume"].where(price_change == 0, 0) * 0.5
        )

        # Efficient rolling calculations
        buy_sum = df["buy_volume"].rolling(volume_bucket_size, min_periods=1).sum()
        sell_sum = df["sell_volume"].rolling(volume_bucket_size, min_periods=1).sum()
        volume_sum = df["volume"].rolling(volume_bucket_size, min_periods=1).sum()

        vpin = abs(buy_sum - sell_sum) / volume_sum

        logger.info("VPIN calculation completed")
        return vpin

    def calculate_hurst_exponent(self, window: int = 50) -> pd.Series:
        """
        Calculate Hurst Exponent for volume clustering detection.
        Optimized version with smaller window and vectorized operations.

        H > 0.5: Trend persistence (institutional activity)
        H < 0.5: Mean reversion
        H = 0.5: Random walk

        Args:
            window (int): Rolling window size for calculation

        Returns:
            pd.Series: Hurst exponent values
        """
        logger.info("Calculating Hurst exponent...")

        # Pre-calculate log returns for volume
        log_volume = np.log(self.df["volume"])

        # Use numpy operations for speed
        def hurst_calc(series):
            # Reduced number of lags for faster computation
            lags = np.array([2, 4, 8, 16])  # Use power of 2 lags for efficiency

            # Normalize series using vectorized operations
            series = (series - np.mean(series)) / np.std(series)

            # Preallocate tau array
            tau = np.zeros(len(lags))

            # Vectorized calculation
            for idx, lag in enumerate(lags):
                # Calculate price difference
                diff = series[lag:] - series[:-lag]
                tau[idx] = np.sqrt(np.std(diff))

            # Avoid log(0) and use vectorized operations
            valid_mask = tau > 0
            if np.sum(valid_mask) < 2:
                return np.nan

            valid_tau = tau[valid_mask]
            valid_lags = lags[valid_mask]

            # Use numpy for polynomial fitting
            poly = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)
            return poly[0] / 2.0 + 0.5

        # Use rolling window with stride for faster computation
        stride = max(1, window // 10)  # Adjust stride based on window size
        hurst_values = []

        for i in range(0, len(log_volume) - window + 1, stride):
            window_data = log_volume.iloc[i : i + window].values
            h = hurst_calc(window_data)
            hurst_values.extend([h] * stride)

        # Fill remaining values
        remaining = len(log_volume) - len(hurst_values)
        if remaining > 0:
            hurst_values.extend([hurst_values[-1]] * remaining)

        return pd.Series(hurst_values, index=self.df.index)

    def calculate_asc(self, window: int = 10) -> pd.Series:
        """
        Calculate Abnormal Spread Contraction (ASC).

        Measures how tight spreads are compared to recent history,
        indicating potential institutional activity.

        Args:
            window (int): Rolling window size in minutes

        Returns:
            pd.Series: ASC values
        """
        logger.info("Calculating ASC...")
        # Calculate approximate spread using high-low
        df = self.df.copy()
        df["spread"] = ((df["high"] - df["low"]) / df["close"]).astype(float)

        # Calculate rolling average spread
        rolling_spread = df["spread"].rolling(window=window).mean()

        # Calculate ASC
        asc = df["spread"] / rolling_spread

        logger.info("ASC calculation completed")
        return asc

    def calculate_vwpd(self, window: int = 10) -> pd.Series:
        """
        Calculate Volume-Weighted Price Dislocation (VWPD).

        Measures price deviation from VWAP, indicating aggressive
        institutional execution.

        Args:
            window (int): Rolling window size in minutes

        Returns:
            pd.Series: VWPD values
        """
        logger.info("Calculating VWPD...")
        df = self.df.copy()

        # Calculate VWAP - ensure float dtype
        df["volume"] = df["volume"].astype(float)
        df["close"] = df["close"].astype(float)

        df["vwap"] = (df["close"] * df["volume"]).rolling(window=window).sum() / df[
            "volume"
        ].rolling(window=window).sum()

        # Calculate VWPD
        vwpd = (df["close"] - df["vwap"]) / df["vwap"]

        logger.info("VWPD calculation completed")
        return vwpd

    def validate_processed_features(self) -> bool:
        """
        Validate processed features against model requirements.

        Checks:
        1. Feature value ranges and distributions
        2. Time series consistency
        3. Cross-correlations
        4. Statistical properties
        5. Volume persistence patterns

        Returns:
            bool: True if validation passes, raises ValueError otherwise
        """
        logger.info("Validating processed features...")

        # 1. VPIN Validation (Volume-Synchronized Probability of Informed Trading)
        if not (0 <= self.df["vpin"].min() <= self.df["vpin"].max() <= 1):
            raise ValueError(
                f"VPIN values out of bounds [0,1]: min={self.df['vpin'].min():.3f}, "
                f"max={self.df['vpin'].max():.3f}"
            )

        # Check VPIN distribution (should have some extreme values for institutional activity)
        vpin_extremes = len(
            self.df[(self.df["vpin"] > 0.7) | (self.df["vpin"] < 0.3)]
        ) / len(self.df)
        if vpin_extremes < 0.1:  # At least 10% should be institutional activity
            logger.warning(
                f"Low institutional activity detected in VPIN "
                f"(only {vpin_extremes:.1%} extreme values)"
            )

        # 2. Hurst Exponent Validation
        hurst_values = self.df["hurst"].dropna()
        if len(hurst_values) == 0:
            raise ValueError("No valid Hurst exponent values found")

        hurst_min, hurst_max = hurst_values.min(), hurst_values.max()
        if not (0 <= hurst_min <= hurst_max <= 1):
            raise ValueError(
                f"Hurst exponent out of bounds [0,1]: min={hurst_min:.3f}, "
                f"max={hurst_max:.3f}"
            )

        # Check for persistence in volume patterns
        hurst_persistence = len(hurst_values[hurst_values > 0.5]) / len(hurst_values)
        logger.info(f"Volume persistence ratio: {hurst_persistence:.2%}")

        # Analyze persistence patterns during high VPIN periods
        high_vpin_mask = self.df["vpin"] > 0.7
        if high_vpin_mask.any():
            high_vpin_hurst = self.df.loc[high_vpin_mask, "hurst"].mean()
            if high_vpin_hurst < 0.5:
                logger.warning(
                    f"Low volume persistence during high VPIN periods: {high_vpin_hurst:.3f}"
                )

        # 3. ASC (Abnormal Spread Contraction) Validation
        asc_mean = self.df["asc"].mean()
        asc_std = self.df["asc"].std()

        if not (0.5 <= asc_mean <= 1.5):
            raise ValueError(
                f"ASC mean value suspicious: {asc_mean:.3f} (expected ~1.0)"
            )

        # Check for institutional patterns in ASC
        asc_institutional = len(self.df[self.df["asc"] < 0.7]) / len(self.df)
        logger.info(f"Potential institutional activity in ASC: {asc_institutional:.2%}")

        # 4. VWPD (Volume-Weighted Price Dislocation) Validation
        if abs(self.df["vwpd"].mean()) > 0.02:  # More than 2% average deviation
            logger.warning(f"VWPD mean seems high: {self.df['vwpd'].mean():.3f}")

        # Check for significant price dislocations
        vwpd_significant = len(self.df[abs(self.df["vwpd"]) > 0.005]) / len(self.df)
        logger.info(f"Significant price dislocations: {vwpd_significant:.2%}")

        # 5. Cross-Feature Validation

        # Check VPIN-ASC correlation (should be negative as tight spreads often accompany informed trading)
        vpin_asc_corr = self.df["vpin"].corr(self.df["asc"])
        if vpin_asc_corr > 0:
            logger.warning(
                f"Unexpected positive correlation between VPIN and ASC: {vpin_asc_corr:.3f}"
            )

        # Check Hurst-VPIN correlation (should be positive during institutional activity)
        hurst_vpin_corr = self.df["hurst"].corr(self.df["vpin"])
        if hurst_vpin_corr < 0:
            logger.warning(
                f"Unexpected negative correlation between Hurst and VPIN: {hurst_vpin_corr:.3f}"
            )

        # 6. Time Series Consistency
        # Check for sudden jumps in features
        for feature in ["vpin", "hurst", "asc", "vwpd"]:
            jumps = abs(self.df[feature].diff()).quantile(0.99)
            if jumps > 0.5:  # Arbitrary threshold for suspicious jumps
                logger.warning(
                    f"Large jumps detected in {feature} "
                    f"(99th percentile of changes: {jumps:.3f})"
                )

        # 7. Data Quality
        # Check for NaN values
        nan_counts = self.df[["vpin", "hurst", "asc", "vwpd"]].isna().sum()
        if nan_counts.any():
            raise ValueError(
                f"Found NaN values in features: {nan_counts[nan_counts > 0]}"
            )

        # Check for infinite values
        inf_counts = np.isinf(self.df[["vpin", "hurst", "asc", "vwpd"]]).sum()
        if inf_counts.any():
            raise ValueError(
                f"Found infinite values in features: {inf_counts[inf_counts > 0]}"
            )

        logger.info("Feature validation completed successfully ✓")
        return True

    def process_features(self) -> pd.DataFrame:
        """
        Process all institutional order flow features.

        Returns:
            pd.DataFrame: DataFrame with all calculated features
        """
        if self.df is None:
            self.df = self.load_data(self.ticker)

        logger.info("Starting feature calculations...")

        # Precompute common values
        self.df["volume"] = self.df["volume"].astype(
            np.float32
        )  # Use float32 for memory efficiency
        self.df["close"] = self.df["close"].astype(np.float32)

        # Calculate features in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all feature calculations
            vpin_future = executor.submit(self.calculate_vpin)
            hurst_future = executor.submit(self.calculate_hurst_exponent)
            asc_future = executor.submit(self.calculate_asc)
            vwpd_future = executor.submit(self.calculate_vwpd)

            # Get results
            self.df["vpin"] = vpin_future.result()
            self.df["hurst"] = hurst_future.result()
            self.df["asc"] = asc_future.result()
            self.df["vwpd"] = vwpd_future.result()

        # Clean up any remaining NaN values efficiently
        self.df = self.df.ffill().fillna(0)

        # Validate processed features
        self.validate_processed_features()

        logger.info(f"Processed features for {self.ticker}")
        logger.info(f"Final data shape: {self.df.shape}")

        return self.df

    def save_features(self) -> None:
        """
        Save processed features to CSV file.
        """
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        output_path = self.output_dir / f"{self.ticker}_features.csv"
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved processed features to {output_path}")


if __name__ == "__main__":
    start_time = time.time()

    # Process features for AAPL first as a test
    ticker = "AAPL"
    # for ticker in DJ_TITANS_50_TICKER:
    try:
        logger.info(f"==========================================")
        logger.info(f"Starting feature engineering for {ticker}")
        logger.info(f"==========================================")

        fe = FeatureEngineering(ticker)
        fe.df = fe.load_data(ticker)
        processed_df = fe.process_features()
        fe.validate_processed_features()
        fe.save_features()

        logger.info(f"Successfully processed {ticker}")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        logger.info(f"Output shape: {processed_df.shape}")
        logger.info(f"==========================================")

    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
