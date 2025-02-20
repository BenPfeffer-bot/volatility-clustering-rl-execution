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
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.config.paths import PROCESSED_DIR, MODELS_DIR
from src.config.settings import DJ_TITANS_50_TICKER
from src.models.tcn_impact import MarketImpactPredictor
from src.utils.log_utils import setup_logging

# Set up logging
logger = setup_logging(__name__)

# Global lock for model saving
save_lock = Lock()


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

    # Calculate percentage of persistent volume patterns
    persistent_ratio = (df["hurst"] > 0.5).mean() * 100
    logger.info(f"Persistent volume patterns ratio: {persistent_ratio:.2f}%")

    # 6. Check for NaN values
    if df.isnull().any().any():
        raise ValueError("Dataset contains NaN values")

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


def verify_model_saved(save_path: Path, model: MarketImpactPredictor) -> bool:
    """
    Verify that the model was saved correctly.

    Args:
        save_path: Path where model should be saved
        model: Trained model instance

    Returns:
        bool: True if model was saved correctly
    """
    if not save_path.exists():
        logger.error(f"Model file not found at {save_path}")
        return False

    try:
        # Try to load the saved model
        test_model = MarketImpactPredictor()
        test_model.load_model(str(save_path))

        # Verify model parameters match
        if (
            test_model.input_size != model.input_size
            or test_model.output_size != model.output_size
            or test_model.num_channels != model.num_channels
        ):
            logger.error("Loaded model parameters do not match original model")
            return False

        logger.info("Model saved and verified successfully")
        return True

    except Exception as e:
        logger.error(f"Error verifying saved model: {str(e)}")
        return False


def train_model(ticker: str, process_id: int) -> Tuple[str, bool]:
    """
    Train TCN model for a given ticker.

    Args:
        ticker: Stock ticker symbol
        process_id: Process identifier for logging

    Returns:
        Tuple of (ticker, success_status)
    """
    logger.info(f"[Process {process_id}] Starting training for {ticker}")
    logger.info("=" * 50)

    try:
        # Load and prepare data
        df = load_and_prepare_data(ticker)

        # Split data
        train_data, val_data = train_test_split(df)

        # Initialize model
        model = MarketImpactPredictor()

        # Train model
        logger.info(f"[Process {process_id}] Starting model training for {ticker}...")
        history = model.train(train_df=train_data, val_df=val_data)

        # Save model with lock to prevent concurrent writes
        save_path = MODELS_DIR / f"tcn_{ticker}.pt"
        with save_lock:
            model.save_model(str(save_path))
            logger.info(f"[Process {process_id}] Model saved at {save_path}")

        # Verify model was saved correctly
        if not verify_model_saved(save_path, model):
            raise RuntimeError(f"Failed to save model for {ticker}")

        logger.info(
            f"[Process {process_id}] Successfully completed training for {ticker}"
        )
        return ticker, True

    except Exception as e:
        logger.error(
            f"[Process {process_id}] Error training model for {ticker}: {str(e)}"
        )
        logger.error("Stack trace:", exc_info=True)
        return ticker, False


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
    """Main training pipeline with improved parallel processing."""
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Get available tickers
    tickers = get_available_tickers()
    # tickers = ["AAPL"]
    logger.info(f"Found {len(tickers)} tickers with processed data")

    # Configure parallel processing
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    logger.info(f"Using {num_workers} workers for parallel training")

    # Track successful and failed models
    successful_models = []
    failed_models = []

    # Process tickers in parallel with process IDs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all training jobs with process IDs
        future_to_ticker = {
            executor.submit(train_model, ticker, i): ticker
            for i, ticker in enumerate(tickers)
        }

        # Process completed jobs with progress bar
        with tqdm(total=len(tickers), desc="Training models") as pbar:
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker, success = future.result()
                    if success:
                        successful_models.append(ticker)
                        logger.info(f"✓ Successfully trained model for {ticker}")
                    else:
                        failed_models.append(ticker)
                        logger.error(f"✗ Failed to train model for {ticker}")
                except Exception as e:
                    failed_models.append(ticker)
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                finally:
                    pbar.update(1)

    # Print summary
    logger.info("\n=== Training Summary ===")
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Successfully trained: {len(successful_models)}")
    logger.info(f"Failed: {len(failed_models)}")

    if failed_models:
        logger.info("\nFailed tickers:")
        for ticker in failed_models:
            logger.info(f"- {ticker}")

    # Clean up
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return appropriate exit code
    return 1 if failed_models else 0


if __name__ == "__main__":
    sys.exit(main())
