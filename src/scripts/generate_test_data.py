"""
Generate test data for backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.config.paths import PROCESSED_DIR
from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


def generate_test_data(
    ticker: str = "AAPL",
    start_date: datetime = datetime(2024, 1, 1),
    days: int = 30,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    Args:
        ticker: Stock ticker
        start_date: Start date for data
        days: Number of days to generate
        freq: Data frequency

    Returns:
        DataFrame with synthetic market data
    """
    # Generate timestamps
    timestamps = pd.date_range(
        start=start_date,
        end=start_date + timedelta(days=days),
        freq=freq,
    )
    timestamps = timestamps[
        (timestamps.hour >= 9) & (timestamps.hour < 16)
    ]  # Market hours

    # Initialize random price process
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.001, len(timestamps))
    price = 100 * np.exp(np.cumsum(returns))

    # Create base DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": price,
            "volume": np.random.lognormal(10, 1, len(timestamps)),
        }
    )
    df.set_index("timestamp", inplace=True)

    # Add features
    df["daily_volatility"] = df["close"].pct_change().rolling(
        window=390
    ).std() * np.sqrt(252)
    df["vpin"] = np.random.uniform(0.3, 0.9, len(df))  # Synthetic VPIN
    df["market_impact_pred"] = np.random.uniform(0.001, 0.02, len(df))

    # Add volume features
    df["avg_volume"] = (
        df["volume"].rolling(window=20).mean()
    )  # 20-period moving average
    df["avg_volume"] = df["avg_volume"].fillna(df["volume"])  # Fill initial NaN values

    # Add regime labels
    regimes = ["institutional_flow", "high_impact", "trending", "neutral"]
    df["regime"] = np.random.choice(regimes, len(df))

    # Clean up data
    df.dropna(inplace=True)

    return df


def main():
    """Generate and save test data."""
    logger.info("Generating test data...")

    # Create test data
    df = generate_test_data()

    # Save to processed directory
    output_file = PROCESSED_DIR / "AAPL_features.csv"
    df.to_csv(output_file)
    logger.info(f"Test data saved to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
