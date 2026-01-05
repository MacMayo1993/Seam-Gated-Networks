"""
Training Loop for Neural Models

Implements training loop with:
- Adam optimizer (lr=1e-3)
- Gradient clipping (max_norm=1.0)
- Early stopping (patience=10)
- MSE loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.

        Args:
            X: Input sequences (N, history_length, input_dim)
            y: Target observations (N, input_dim)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if should stop training.

        Args:
            val_loss: Validation loss
            model: Model being trained

        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"  Validation loss improved: {self.best_loss:.6f} -> {val_loss:.6f}")
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0

        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()

        # Handle different model interfaces
        if hasattr(model, 'forward'):
            # For Z2 models that may return diagnostics
            output = model(X_batch)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output
        else:
            predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate model.

    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            output = model(X_batch)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output

            # Compute loss
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Train a neural network model.

    Args:
        model: Neural network model
        X_train: Training input sequences (N_train, history_length, input_dim)
        y_train: Training targets (N_train, input_dim)
        X_val: Validation input sequences (N_val, history_length, input_dim)
        y_val: Validation targets (N_val, input_dim)
        epochs: Maximum number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for Adam optimizer
        patience: Early stopping patience
        device: Device to train on (default: cuda if available, else cpu)
        verbose: Whether to print training progress

    Returns:
        Dictionary with training history and best model state
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Training loop
    if verbose:
        print(f"Training on {device}")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Early stopping check
        if early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Restore best model
    model.load_state_dict(early_stopping.best_model_state)

    history['best_val_loss'] = early_stopping.best_loss
    history['epochs_trained'] = epoch + 1

    return history


if __name__ == '__main__':
    # Test training loop
    print("Testing Training Loop")
    print("=" * 60)

    from models.lstm import StandardLSTM

    # Create dummy data
    N_train = 1000
    N_val = 200
    history_length = 10
    input_dim = 4

    X_train = np.random.randn(N_train, history_length, input_dim)
    y_train = np.random.randn(N_train, input_dim)
    X_val = np.random.randn(N_val, history_length, input_dim)
    y_val = np.random.randn(N_val, input_dim)

    # Create model
    model = StandardLSTM(input_dim=input_dim, hidden_dim=32)

    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=32,
        verbose=True
    )

    print(f"\nTraining completed:")
    print(f"  Epochs trained: {history['epochs_trained']}")
    print(f"  Best val loss: {history['best_val_loss']:.6f}")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")

    print("\nTest passed!")
