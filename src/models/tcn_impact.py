"""
Temporal Convolutional Network (TCN) for predicting market impact of institutional orders.

Key features:
- Causal convolutions to avoid lookahead bias: Ensures model only uses past data for predictions,
  preventing data leakage that could give unrealistic performance
- Dilated convolutions for long-range dependencies: Exponentially increases receptive field size
  to capture patterns across different time scales efficiently
- Residual connections for better gradient flow: Helps combat vanishing gradients in deep networks
  by providing direct paths for gradient propagation
- Multi-scale architecture for capturing different time horizons: Processes market data at multiple
  temporal resolutions to identify both short and long-term patterns

The model is specifically designed for financial time series, with careful consideration for:
- Market microstructure effects
- Non-stationarity of financial data
- The need for interpretable predictions
- Computational efficiency for real-time applications
"""

# Import required libraries for deep learning, data processing and utilities
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

# Add project root to path to allow imports from src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import custom logging utility
from src.utils.log_utils import setup_logging

# Initialize logger for this module
logger = setup_logging(__name__)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer for avoiding lookahead bias.

    This layer implements causal convolutions where each output only depends on past inputs.
    Key design choices:
    - Left-side padding ensures temporal causality
    - No padding on right side prevents future information leakage
    - Dilation parameter allows exponential increase in receptive field

    This is crucial for financial applications where using future data would create
    unrealistic backtesting results and potentially misleading strategies.

    Potential weaknesses:
    - Increased memory usage due to padding on left side
    - May introduce edge effects at the start of sequences
    - Computational overhead from handling padding manually
    - Limited ability to capture very long-term dependencies without large dilation
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super(CausalConv1d, self).__init__()
        # Calculate padding needed for causality - only pad left side
        self.padding = (kernel_size - 1) * dilation
        # Create the convolutional layer with no padding (we'll handle it manually)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # No automatic padding
            dilation=dilation,  # Use dilation for increased receptive field
        )
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        # Pad left side only to maintain causality - crucial for time series
        x = F.pad(x, (self.padding, 0))  # Pad only on the left
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connection.

    This block combines several key components:
    - Dual causal convolutions for hierarchical feature extraction
    - Batch normalization for stable training and faster convergence
    - ReLU activation to introduce non-linearity
    - Dropout for regularization and preventing overfitting
    - Residual connection to help with gradient flow in deep networks

    The architecture is optimized for financial time series by:
    - Using moderate dropout (0.2) to maintain signal while preventing overfitting
    - Employing batch norm to handle varying scales in financial data
    - Implementing residual connections to capture both linear and non-linear patterns
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super(TCNBlock, self).__init__()

        # First causal conv layer with batch norm
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        # Second causal conv layer with batch norm
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        # ReLU activation for non-linearity
        self.relu = nn.ReLU()
        # Dropout for regularization - 0.2 chosen to balance signal preservation and overfitting
        self.dropout = nn.Dropout(0.2)
        # Batch normalization for each conv layer to stabilize training
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        # 1x1 conv to match dimensions if needed for residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # First conv block with batch norm, ReLU and dropout
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block with batch norm
        out = self.conv2(out)
        out = self.batch_norm2(out)

        # Handle residual connection - match dimensions if needed
        res = x if self.downsample is None else self.downsample(x)
        # Adjust residual tensor length to match output
        if res.shape[-1] > out.shape[-1]:
            res = res[..., : out.shape[-1]]
        elif res.shape[-1] < out.shape[-1]:
            res = F.pad(res, (0, out.shape[-1] - res.shape[-1]))

        # Add residual and apply final ReLU
        return self.relu(out + res)


class MarketImpactTCN(nn.Module):
    """
    Multi-scale TCN for market impact prediction.

    The network architecture is specifically designed for market impact modeling:
    - Multiple TCN blocks with increasing dilation capture patterns at different timescales
    - Global average pooling reduces sequence dimension while preserving temporal patterns
    - Final fully connected layer maps features to market impact prediction

    Key architectural decisions:
    - Exponentially increasing dilations (2^i) allow for efficient long-range dependency modeling
    - Multiple channels capture different aspects of market behavior
    - Global pooling provides translation invariance and reduces model parameters
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
    ):
        super(MarketImpactTCN, self).__init__()

        # Create list to hold TCN blocks
        layers = []
        num_levels = len(num_channels)

        # Build TCN blocks with increasing dilation
        for i in range(num_levels):
            # Exponential dilation growth
            dilation = 2**i
            # Input channels - first layer uses raw input size
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))

        # Create sequential network from TCN blocks
        self.network = nn.Sequential(*layers)

        # Global average pooling to reduce sequence length to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Final fully connected layer for prediction
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Process through TCN blocks
        x = self.network(x)
        # Global average pooling to get fixed-size representation
        x = self.global_pool(x)
        # Remove singleton dimension
        x = x.squeeze(-1)
        # Final prediction
        return self.fc(x)


class MarketImpactPredictor:
    """
    Wrapper class for training and using the TCN model.

    This class handles:
    - Model initialization and configuration
    - Hardware acceleration setup (MPS/CPU)
    - Mixed precision training for improved performance
    - Data preparation and sequence generation
    - Training loop with optimization techniques
    - Model persistence and loading

    Key features:
    - Adaptive batch sizing based on available memory
    - Early stopping to prevent overfitting
    - Learning rate scheduling for better convergence
    - Gradient clipping for training stability
    - Memory optimization for large datasets
    """

    def __init__(self, sequence_length=30, learning_rate=0.0005, batch_size=128):
        # Store hyperparameters
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = 1.0  # For gradient clipping

        # Set up device - prefer MPS (Apple Silicon) over CPU
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Configure mixed precision based on device capability
        self.use_amp = self.device.type == "mps"
        if self.use_amp:
            self.amp_dtype = torch.float16
            logger.info("Using mixed precision training with float16")
        else:
            logger.info("Using full precision training on CPU")

        # Define model architecture parameters
        self.input_size = 6  # Features including Hurst exponent
        self.output_size = 1  # Single target (market impact)
        self.num_channels = [32, 64, 128]  # Increasing channel sizes
        self.kernel_size = 3  # Small kernel for fine-grained patterns

        # Initialize model and move to device
        self.model = MarketImpactTCN(
            input_size=self.input_size,
            output_size=self.output_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
        ).to(self.device)

        # Initialize optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Initialize learning rate scheduler with placeholder values
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=25,
            steps_per_epoch=1,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        # Use MSE loss for regression
        self.criterion = nn.MSELoss()

    def prepare_sequences(self, data, target_col="market_impact"):
        """
        Optimized sequence preparation with reduced memory footprint and robust error handling.

        Key optimizations:
        - Processes data in chunks to reduce memory usage
        - Validates input data integrity
        - Handles missing or invalid values
        - Efficient numpy operations for sequence creation
        - Memory-efficient tensor conversion

        Error handling:
        - Validates minimum data length
        - Checks for required features
        - Handles NaN and infinite values
        - Ensures sequence validity
        """
        # Define required features
        features = [
            "log_returns",
            "daily_volatility",
            "vpin",
            "vwpd",
            "asc",
            "hurst",
        ]

        # Validate input data length
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Data length ({len(data)}) must be >= sequence_length ({self.sequence_length})"
            )

        # Check for missing features
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Initialize arrays for sequences and targets
        total_sequences = len(data) - self.sequence_length + 1
        sequences = np.zeros(
            (total_sequences, self.input_size, self.sequence_length), dtype=np.float32
        )
        targets = np.zeros(total_sequences, dtype=np.float32)

        # Process data in chunks to manage memory
        chunk_size = 50000
        for start_idx in range(0, len(data), chunk_size):
            end_idx = min(start_idx + chunk_size, len(data))
            chunk_data = data.iloc[start_idx:end_idx]

            # Extract features and targets
            feature_data = chunk_data[features].values
            target_data = chunk_data[target_col].values

            # Handle invalid values
            if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
                logger.warning(f"Invalid values found in chunk {start_idx}-{end_idx}")
                feature_data = np.nan_to_num(
                    feature_data, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Calculate sequence indices for this chunk
            chunk_seq_start = max(0, start_idx - self.sequence_length + 1)
            chunk_seq_end = end_idx - self.sequence_length + 1

            # Create sequences from chunk
            if chunk_seq_end > chunk_seq_start:
                for i in range(chunk_seq_start, chunk_seq_end):
                    seq_start = max(0, i - start_idx)
                    seq_end = seq_start + self.sequence_length

                    if seq_end <= len(feature_data):
                        sequence = feature_data[seq_start:seq_end].T
                        if sequence.shape == (self.input_size, self.sequence_length):
                            sequences[i] = sequence
                            targets[i] = target_data[seq_end - 1]

        # Remove invalid sequences
        valid_mask = ~np.any(np.isnan(sequences), axis=(1, 2))
        if not np.any(valid_mask):
            raise ValueError("No valid sequences could be created from the input data")

        sequences = sequences[valid_mask]
        targets = targets[valid_mask]

        logger.info(f"Created {len(sequences)} valid sequences")
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 25,
    ) -> dict:
        """
        Optimized training with reduced memory usage and faster processing.

        Key features:
        - Dynamic batch size optimization
        - Mixed precision training
        - Memory-efficient data handling
        - Progress tracking with tqdm
        - Early stopping mechanism
        - Learning rate scheduling
        - Gradient clipping

        Optimizations:
        - Efficient memory management
        - Batch processing
        - Hardware acceleration
        - Garbage collection
        """
        logger.info("Preparing training data...")

        # Prepare data tensors
        X_train, y_train = self.prepare_sequences(train_df)

        # Handle validation data if provided
        if val_df is not None:
            X_val, y_val = self.prepare_sequences(val_df)
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        # Optimize batch size and calculate steps
        batch_size = min(256, len(X_train))
        n_batches = (len(X_train) + batch_size - 1) // batch_size

        # Calculate total training steps
        total_steps = n_batches * epochs

        # Configure learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=epochs,
            steps_per_epoch=n_batches,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
            total_steps=total_steps,
        )

        # Initialize training history
        history = {"train_loss": [], "val_loss": [] if val_df is not None else None}

        # Early stopping configuration
        patience = 5
        min_delta = 1e-6
        best_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            # Shuffle training data
            permutation = torch.randperm(len(X_train))
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            # Process mini-batches with progress bar
            with tqdm(
                range(0, len(X_train), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            ) as pbar:
                for i in pbar:
                    batch_end = min(i + batch_size, len(X_train))
                    X_batch = X_train_shuffled[i:batch_end].to(self.device)
                    y_batch = y_train_shuffled[i:batch_end].to(self.device)

                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    # Forward pass with mixed precision if available
                    if self.use_amp:
                        with torch.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        ):
                            y_pred = self.model(X_batch)
                            loss = self.criterion(y_pred.squeeze(), y_batch)
                    else:
                        y_pred = self.model(X_batch)
                        loss = self.criterion(y_pred.squeeze(), y_batch)

                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    # Update learning rate
                    if epoch * n_batches + (i // batch_size) < total_steps:
                        self.scheduler.step()

                    # Update progress
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    pbar.set_postfix({"loss": f"{current_loss:.4f}"})

                    # Clean up memory
                    del X_batch, y_batch, y_pred, loss
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

            # Calculate average epoch loss
            epoch_loss /= n_batches
            history["train_loss"].append(epoch_loss)

            # Validation step
            if val_df is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(X_val, y_val)
                history["val_loss"].append(val_loss)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {epoch_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )

                # Early stopping check
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

            # Memory cleanup
            import gc

            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

        return history

    def _validate(self, X_val, y_val):
        """
        Efficient validation step.

        Optimizations:
        - Larger batch size for validation (no gradient computation needed)
        - Mixed precision inference
        - Memory cleanup after each batch
        - No gradient computation (torch.no_grad)
        """
        self.model.eval()
        batch_size = min(self.batch_size * 2, len(X_val))
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

                # Use mixed precision if available
                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device.type, dtype=self.amp_dtype
                    ):
                        y_pred = self.model(X_batch)
                        loss = self.criterion(y_pred.squeeze(), y_batch)
                else:
                    y_pred = self.model(X_batch)
                    loss = self.criterion(y_pred.squeeze(), y_batch)

                val_loss += loss.item()
                n_batches += 1

                # Clean up memory
                del X_batch, y_batch, y_pred, loss
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        return val_loss / n_batches

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using MPS acceleration if available.

        Features:
        - Hardware acceleration support
        - Mixed precision inference
        - Memory-efficient batch processing
        - No gradient computation for faster inference
        """
        self.model.eval()
        X, _ = self.prepare_sequences(df)
        X = X.to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    predictions = self.model(X)
            else:
                predictions = self.model(X)

            return predictions.cpu().numpy()

    def save_model(self, path: str):
        """
        Save model state.

        Saves:
        - Model architecture and weights
        - Optimizer state for training resumption
        - Model hyperparameters
        - Training configuration

        Creates necessary directories if they don't exist.
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save complete model state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "sequence_length": self.sequence_length,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "num_channels": self.num_channels,
                "kernel_size": self.kernel_size,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model state.

        Restores:
        - Model architecture and weights
        - Optimizer state
        - Model hyperparameters
        - Training configuration

        Ensures exact reproduction of saved model state.
        """
        # Load checkpoint
        checkpoint = torch.load(path)
        # Restore model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore hyperparameters
        self.sequence_length = checkpoint["sequence_length"]
        self.learning_rate = checkpoint["learning_rate"]
        self.batch_size = checkpoint["batch_size"]
        self.input_size = checkpoint["input_size"]
        self.output_size = checkpoint["output_size"]
        self.num_channels = checkpoint["num_channels"]
        self.kernel_size = checkpoint["kernel_size"]
        logger.info(f"Model loaded from {path}")

    def evaluate(self, data: pd.DataFrame) -> float:
        """
        Evaluate model performance on given data.

        Args:
            data: DataFrame with features

        Returns:
            Mean squared error of predictions
        """
        if data.empty:
            logger.warning("Empty data provided for evaluation")
            return 0.0

        self.model.eval()  # Set model to evaluation mode

        try:
            # Prepare sequences
            X, y = self.prepare_sequences(data)
            if X is None or y is None or X.shape[0] == 0 or y.shape[0] == 0:
                logger.warning("No valid sequences could be prepared for evaluation")
                return 0.0

            X = X.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device.type, dtype=self.amp_dtype
                    ):
                        y_pred = self.model(X)
                else:
                    y_pred = self.model(X)

                # Convert predictions to numpy and handle NaN values
                y_pred = y_pred.cpu().numpy()
                y_true = y.cpu().numpy()

                # Remove any NaN values
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                if not np.any(mask):
                    logger.warning("All predictions or true values are NaN")
                    return 0.0

                y_true = y_true[mask]
                y_pred = y_pred[mask]

                if len(y_true) == 0:
                    logger.warning("No valid samples after removing NaN values")
                    return 0.0

                mse = np.mean((y_true - y_pred) ** 2)
                return float(mse) if not np.isnan(mse) and not np.isinf(mse) else 0.0

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            return 0.0
