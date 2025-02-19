"""
Train the TCN model for market impact prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config.paths import PROCESSED_DIR, MODELS_DIR
from src.config.settings import DJ_TITANS_50_TICKER
from src.models.tcn_impact import MarketImpactPredictor
from src.utils.log_utils import setup_logging

# Set up logging
logger = setup_logging(__name__)


def get_available_tickers():
    """Get list of tickers that have processed feature files."""
    feature_files = list(PROCESSED_DIR.glob("*_features.csv"))
    return [f.stem.replace("_features", "") for f in feature_files]


def validate_features(df: pd.DataFrame) -> bool:
    """
    Validate features for TCN model training.

    Checks:
    1. Required features presence
    2. VPIN bounds (should be between 0 and 1)
    3. ASC quality (relative to rolling spread)
    4. VWPD reasonableness
    5. Hurst exponent bounds and persistence patterns
    6. Data completeness and NaN checks

    Args:
        df: DataFrame with features

    Returns:
        bool: True if validation passes, raises ValueError otherwise
    """
    logger.info("Validating features...")

    # 1. Check required features
    required_features = {
        "log_returns": "Price movement feature",
        "daily_volatility": "Volatility context",
        "vpin": "Volume-Synchronized Probability of Informed Trading",
        "vwpd": "Volume-Weighted Price Dislocation",
        "asc": "Abnormal Spread Contraction",
        "hurst": "Volume clustering persistence",
    }

    missing = [feat for feat in required_features if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing critical features: {missing}")

    # 2. Validate VPIN bounds
    if not (0 <= df["vpin"].min() <= df["vpin"].max() <= 1):
        raise ValueError(
            f"VPIN values out of bounds [0,1]: min={df['vpin'].min():.3f}, max={df['vpin'].max():.3f}"
        )

    # 3. Validate ASC (should mostly be around 1.0 with deviations)
    asc_mean = df["asc"].mean()
    if not (0.5 <= asc_mean <= 1.5):
        raise ValueError(f"ASC mean value suspicious: {asc_mean:.3f} (expected ~1.0)")

    # 4. Check VWPD reasonableness (should be small percentage differences)
    if abs(df["vwpd"].mean()) > 0.02:  # More than 2% average deviation is suspicious
        logger.warning(
            f"VWPD mean seems high: {df['vwpd'].mean():.3f} (>2% average deviation)"
        )

    # 5. Validate Hurst exponent
    hurst_min, hurst_max = df["hurst"].min(), df["hurst"].max()
    if not (0 <= hurst_min <= hurst_max <= 1):
        raise ValueError(
            f"Hurst exponent out of bounds [0,1]: min={hurst_min:.3f}, max={hurst_max:.3f}"
        )

    # Check for persistence patterns
    persistent_ratio = len(df[df["hurst"] > 0.5]) / len(df)
    logger.info(f"Persistent volume patterns ratio: {persistent_ratio:.2%}")
    if persistent_ratio < 0.3:  # Less than 30% showing persistence is unusual
        logger.warning("Low persistence in volume patterns detected")

    # 6. Check for data quality
    nan_counts = df[list(required_features.keys())].isna().sum()
    if nan_counts.any():
        raise ValueError(f"Found NaN values in features: {nan_counts[nan_counts > 0]}")

    # Check for infinite values
    inf_counts = np.isinf(df[list(required_features.keys())]).sum()
    if inf_counts.any():
        raise ValueError(
            f"Found infinite values in features: {inf_counts[inf_counts > 0]}"
        )

    logger.info("Feature validation passed ✓")
    return True


def load_and_prepare_data(ticker: str) -> pd.DataFrame:
    """
    Load processed features and prepare data for training.

    Args:
        ticker: Stock ticker symbol

    Returns:
        DataFrame with features and target variables
    """
    logger.info(f"Loading data for {ticker}...")
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}_features.csv")
    logger.info(f"Loaded {len(df)} rows of data")

    # Calculate target variable (market impact)
    logger.info("Calculating target variables...")
    df["market_impact"] = df["close"].pct_change().shift(-1)
    df = df.dropna()

    # Validate features before proceeding
    validate_features(df)

    logger.info(f"Final data shape after preprocessing: {df.shape}")
    return df


def train_test_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.

    Args:
        df: DataFrame with features and target
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (training_data, validation_data)
    """
    split_idx = int(len(df) * train_ratio)
    train_data = df.iloc[:split_idx]
    val_data = df.iloc[split_idx:]

    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {val_data.shape}")

    return train_data, val_data


def train_model(ticker: str) -> MarketImpactPredictor:
    """
    Train TCN model for a given ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Trained MarketImpactPredictor instance
    """
    logger.info("=" * 50)
    logger.info(f"Training model for {ticker}")
    logger.info("=" * 50)

    try:
        # Load and prepare data
        df = load_and_prepare_data(ticker)

        # Split data
        train_data, val_data = train_test_split(df)

        # Initialize model
        model = MarketImpactPredictor()

        # Prepare sequences
        logger.info("Preparing sequences...")
        X_train, y_train = model.prepare_sequences(train_data)
        X_val, y_val = model.prepare_sequences(val_data)

        logger.info(f"Training sequences shape: {X_train.shape}")
        logger.info(f"Validation sequences shape: {X_val.shape}")

        # Train model
        logger.info("Starting model training...")
        history = model.train(train_df=train_data, val_df=val_data)

        # Save model
        save_path = MODELS_DIR / f"tcn_{ticker}.pt"
        model.save_model(save_path)
        logger.info(f"Model saved to {save_path}")

        return model

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


def evaluate_predictions(model: MarketImpactPredictor, test_df: pd.DataFrame) -> dict:
    """
    Evaluate model predictions.

    Args:
        model: Trained MarketImpactPredictor instance
        test_df: Test data DataFrame

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model predictions...")
    predictions = model.predict(test_df)

    # Calculate metrics
    metrics = {
        "price_impact_mae": np.mean(np.abs(predictions[:, 0])),
        "volume_impact_mae": np.mean(np.abs(predictions[:, 1])),
        "price_impact_rmse": np.sqrt(np.mean(predictions[:, 0] ** 2)),
        "volume_impact_rmse": np.sqrt(np.mean(predictions[:, 1] ** 2)),
    }

    return metrics


def main():
    """Main training pipeline with parallel processing."""
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Get available tickers
    tickers = get_available_tickers()
    logger.info(f"Found {len(tickers)} tickers with processed data")

    # Configure parallel processing
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    # Use 75% of available CPU cores for training
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    logger.info(f"Using {num_workers} workers for parallel training")

    # Process tickers in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all training jobs
        future_to_ticker = {
            executor.submit(train_model, ticker): ticker for ticker in tickers
        }

        # Process completed jobs
        from concurrent.futures import as_completed

        for future in tqdm(
            as_completed(future_to_ticker), total=len(tickers), desc="Training models"
        ):
            ticker = future_to_ticker[future]
            try:
                model = future.result()
                logger.info(f"Successfully trained model for {ticker}")
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                continue


if __name__ == "__main__":
    main()
