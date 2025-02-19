"""
Temporal Convolutional Network (TCN) for predicting market impact of institutional orders.

Key features:
- Causal convolutions to avoid lookahead bias
- Dilated convolutions for long-range dependencies
- Residual connections for better gradient flow
- Multi-scale architecture for capturing different time horizons
"""

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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.log_utils import setup_logging

logger = setup_logging(__name__)


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer for avoiding lookahead bias.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Pad only on the left
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super(TCNBlock, self).__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        # 1x1 conv to match dimensions if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # Input shape: (batch, channels, sequence_length)
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        res = x if self.downsample is None else self.downsample(x)
        if res.shape[-1] > out.shape[-1]:
            res = res[..., : out.shape[-1]]
        elif res.shape[-1] < out.shape[-1]:
            res = F.pad(res, (0, out.shape[-1] - res.shape[-1]))

        return self.relu(out + res)


class MarketImpactTCN(nn.Module):
    """
    Multi-scale TCN for market impact prediction.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
    ):
        super(MarketImpactTCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))

        self.network = nn.Sequential(*layers)

        # Global average pooling followed by fully connected layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch_size, input_size, sequence_length)
        x = self.network(x)
        # Global average pooling
        x = self.global_pool(x)
        # Reshape: (batch_size, channels, 1) -> (batch_size, channels)
        x = x.squeeze(-1)
        # Final prediction
        return self.fc(x)


class MarketImpactPredictor:
    """
    Wrapper class for training and using the TCN model.
    """

    def __init__(self, sequence_length=30, learning_rate=0.0005, batch_size=128):
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = 1.0

        # Configure device - Use MPS if available, fallback to CPU
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        logger.info(f"Using device: {self.device}")

        # Configure mixed precision based on device
        self.use_amp = self.device.type == "mps"
        if self.use_amp:
            self.amp_dtype = torch.float16
            logger.info("Using mixed precision training with float16")
        else:
            logger.info("Using full precision training on CPU")

        # Model parameters
        self.input_size = 6  # Updated from 5 to include Hurst exponent
        self.output_size = 1
        self.num_channels = [32, 64, 128]
        self.kernel_size = 3

        self.model = MarketImpactTCN(
            input_size=self.input_size,
            output_size=self.output_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
        ).to(self.device)

        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01
        )

        # Initialize scheduler with dummy values (will be updated in train())
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=100,
            steps_per_epoch=1,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        self.criterion = nn.MSELoss()

    def prepare_sequences(self, data, target_col="market_impact"):
        """Optimized sequence preparation with reduced memory footprint and robust error handling."""
        features = [
            "log_returns",
            "daily_volatility",
            "vpin",
            "vwpd",
            "asc",
            "hurst",
        ]

        # Validate input data
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Data length ({len(data)}) must be >= sequence_length ({self.sequence_length})"
            )

        # Ensure all features are present
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Convert to numpy and preprocess in chunks
        chunk_size = 50000  # Process data in smaller chunks
        total_sequences = len(data) - self.sequence_length + 1
        sequences = np.zeros(
            (total_sequences, self.input_size, self.sequence_length), dtype=np.float32
        )
        targets = np.zeros(total_sequences, dtype=np.float32)

        for start_idx in range(0, len(data), chunk_size):
            end_idx = min(start_idx + chunk_size, len(data))
            chunk_data = data.iloc[start_idx:end_idx]

            # Process chunk
            feature_data = chunk_data[features].values
            target_data = chunk_data[target_col].values

            # Validate feature data
            if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
                logger.warning(f"Invalid values found in chunk {start_idx}-{end_idx}")
                feature_data = np.nan_to_num(
                    feature_data, nan=0.0, posinf=0.0, neginf=0.0
                )

            # Calculate valid sequences for this chunk
            chunk_seq_start = max(0, start_idx - self.sequence_length + 1)
            chunk_seq_end = end_idx - self.sequence_length + 1

            if chunk_seq_end > chunk_seq_start:
                for i in range(chunk_seq_start, chunk_seq_end):
                    seq_start = max(0, i - start_idx)
                    seq_end = seq_start + self.sequence_length

                    if seq_end <= len(feature_data):
                        # Extract sequence and transpose to (features, sequence_length)
                        sequence = feature_data[seq_start:seq_end].T
                        if sequence.shape == (self.input_size, self.sequence_length):
                            sequences[i] = sequence
                            targets[i] = target_data[seq_end - 1]

        # Validate final sequences
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
        epochs: int = 100,
    ) -> dict:
        """Optimized training with reduced memory usage and faster processing."""
        logger.info("Preparing training data...")

        # Convert data to tensors with optimal memory usage
        X_train, y_train = self.prepare_sequences(train_df)

        # Pre-allocate validation data if available
        if val_df is not None:
            X_val, y_val = self.prepare_sequences(val_df)
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        # Optimize batch size based on available memory
        batch_size = min(256, len(X_train))  # Increased default batch size
        n_batches = len(X_train) // batch_size

        # Update scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=epochs,
            steps_per_epoch=n_batches,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4,
        )

        history = {"train_loss": [], "val_loss": [] if val_df is not None else None}

        # Training loop with optimizations
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            # Create batches efficiently
            permutation = torch.randperm(len(X_train))
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            # Process mini-batches
            with tqdm(
                range(0, len(X_train), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
            ) as pbar:
                for i in pbar:
                    batch_end = min(i + batch_size, len(X_train))
                    X_batch = X_train_shuffled[i:batch_end].to(self.device)
                    y_batch = y_train_shuffled[i:batch_end].to(self.device)

                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)

                    # Forward pass with mixed precision
                    if self.use_amp:
                        with torch.autocast(
                            device_type=self.device.type, dtype=self.amp_dtype
                        ):
                            y_pred = self.model(X_batch)
                            loss = self.criterion(y_pred.squeeze(), y_batch)
                    else:
                        y_pred = self.model(X_batch)
                        loss = self.criterion(y_pred.squeeze(), y_batch)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
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

            # Validation step with reduced frequency
            if val_df is not None and (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_loss = self._validate(X_val, y_val)
                history["val_loss"].append(val_loss)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {epoch_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

            # Force garbage collection between epochs
            import gc

            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

        return history

    def _validate(self, X_val, y_val):
        """Efficient validation step."""
        self.model.eval()
        batch_size = min(
            self.batch_size * 2, len(X_val)
        )  # Larger batch size for validation
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i : i + batch_size]
                y_batch = y_val[i : i + batch_size]

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

                del X_batch, y_batch, y_pred, loss
                if self.device.type == "mps":
                    torch.mps.empty_cache()

        return val_loss / n_batches

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using MPS acceleration if available."""
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

        Args:
            path: Path to save the model state
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state
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

        Args:
            path: Path to load the model state from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.sequence_length = checkpoint["sequence_length"]
        self.learning_rate = checkpoint["learning_rate"]
        self.batch_size = checkpoint["batch_size"]
        self.input_size = checkpoint["input_size"]
        self.output_size = checkpoint["output_size"]
        self.num_channels = checkpoint["num_channels"]
        self.kernel_size = checkpoint["kernel_size"]
        logger.info(f"Model loaded from {path}")
