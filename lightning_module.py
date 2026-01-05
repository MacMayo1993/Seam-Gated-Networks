"""
PyTorch Lightning Module for Seam-Gated Networks

Production-grade training framework with:
- Automatic mixed precision (AMP)
- Learning rate scheduling (OneCycleLR)
- Gradient clipping
- Logging and checkpointing
- Multi-GPU support
- Efficient data loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Import models
from models.lstm import StandardLSTM
from models.z2_equivariant import Z2EquivariantRNN
from models.z2_seam_gated import Z2SeamGatedRNN
from models.gated_delta import Z2GatedDeltaRNN


class SeamGatedLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for Seam-Gated Networks.

    Features:
    - Automatic mixed precision training
    - OneCycleLR scheduling
    - Gradient clipping
    - Comprehensive logging
    - Model checkpointing
    """

    def __init__(
        self,
        model_type: str = 'z2_seam',
        input_dim: int = 4,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 50,
        warmup_epochs: int = 5,
        gradient_clip_val: float = 1.0,
        use_amp: bool = True,
        **model_kwargs
    ):
        """
        Initialize Lightning module.

        Args:
            model_type: Model architecture ('lstm', 'z2_equiv', 'z2_seam', 'z2_delta')
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            learning_rate: Peak learning rate for OneCycleLR
            weight_decay: L2 regularization strength
            max_epochs: Total training epochs
            warmup_epochs: Warmup epochs for learning rate
            gradient_clip_val: Gradient clipping threshold
            use_amp: Use automatic mixed precision
            **model_kwargs: Additional model-specific arguments
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Create model
        self.model = self._create_model(
            model_type, input_dim, hidden_dim, **model_kwargs
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _create_model(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        **kwargs
    ) -> nn.Module:
        """Create model based on type."""
        models = {
            'lstm': StandardLSTM,
            'z2_equiv': Z2EquivariantRNN,
            'z2_seam': Z2SeamGatedRNN,
            'z2_delta': Z2GatedDeltaRNN
        }

        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = models[model_type]

        # Filter kwargs for this model
        import inspect
        sig = inspect.signature(model_class.__init__)
        valid_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters
        }

        return model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **valid_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Handle different model interfaces
        output = self.model(x)

        if isinstance(output, tuple):
            return output[0]  # Return predictions only
        return output

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch

        # Forward pass with automatic mixed precision
        predictions = self(x)

        # Compute loss
        loss = self.criterion(predictions, y)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/mse', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch

        # Forward pass
        predictions = self(x)

        # Compute loss
        loss = self.criterion(predictions, y)

        # Store for epoch-end processing
        self.validation_step_outputs.append({
            'val_loss': loss,
            'predictions': predictions.detach(),
            'targets': y.detach()
        })

        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        outputs = self.validation_step_outputs

        # Average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Compute additional metrics
        all_preds = torch.cat([x['predictions'] for x in outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in outputs], dim=0)

        mse = F.mse_loss(all_preds, all_targets)
        mae = F.l1_loss(all_preds, all_targets)

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/mse', mse)
        self.log('val/mae', mae)

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch

        # Forward pass
        predictions = self(x)

        # Compute loss
        loss = self.criterion(predictions, y)

        # Store for epoch-end processing
        self.test_step_outputs.append({
            'test_loss': loss,
            'predictions': predictions.detach(),
            'targets': y.detach()
        })

        return {'test_loss': loss}

    def on_test_epoch_end(self):
        """Aggregate test metrics."""
        outputs = self.test_step_outputs

        # Average loss
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        # Compute metrics
        all_preds = torch.cat([x['predictions'] for x in outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in outputs], dim=0)

        mse = F.mse_loss(all_preds, all_targets)
        mae = F.l1_loss(all_preds, all_targets)

        # Log metrics
        self.log('test/loss', avg_loss)
        self.log('test/mse', mse)
        self.log('test/mae', mae)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Uses:
        - Adam optimizer with weight decay
        - OneCycleLR for super-convergence
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        # OneCycleLR: Warm up to max_lr, then decay
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=self.hparams.warmup_epochs / self.hparams.max_epochs,
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / div_factor
            final_div_factor=1e4  # min_lr = initial_lr / final_div_factor
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None
    ):
        """
        Configure gradient clipping.

        Uses norm-based clipping for stability.
        """
        if gradient_clip_val is None:
            gradient_clip_val = self.hparams.gradient_clip_val

        if gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm='norm'
            )


class SeamGatedDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for efficient data loading.

    Features:
    - Pin memory for faster GPU transfer
    - Multi-worker data loading
    - Automatic train/val/test splits
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize DataModule.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
        """
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from torch.utils.data import TensorDataset

        if stage == 'fit' or stage is None:
            self.train_dataset = TensorDataset(
                torch.FloatTensor(self.X_train),
                torch.FloatTensor(self.y_train)
            )
            self.val_dataset = TensorDataset(
                torch.FloatTensor(self.X_val),
                torch.FloatTensor(self.y_val)
            )

        if stage == 'test' or stage is None:
            self.test_dataset = TensorDataset(
                torch.FloatTensor(self.X_test),
                torch.FloatTensor(self.y_test)
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )


def create_trainer(
    max_epochs: int = 50,
    accelerator: str = 'auto',
    devices: int = 1,
    precision: str = '16-mixed',
    log_dir: str = 'lightning_logs',
    enable_checkpointing: bool = True,
    enable_early_stopping: bool = True,
    patience: int = 10
) -> pl.Trainer:
    """
    Create configured PyTorch Lightning Trainer.

    Args:
        max_epochs: Maximum training epochs
        accelerator: Device type ('auto', 'gpu', 'cpu', 'tpu')
        devices: Number of devices
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        log_dir: Logging directory
        enable_checkpointing: Save model checkpoints
        enable_early_stopping: Enable early stopping
        patience: Early stopping patience

    Returns:
        Configured Trainer
    """
    callbacks = []

    # Model checkpointing
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{log_dir}/checkpoints',
            filename='seam-gated-{epoch:02d}-{val/loss:.6f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    # Early stopping
    if enable_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Logger
    logger = TensorBoardLogger(log_dir, name='seam_gated')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        deterministic=False,  # Set to True for reproducibility (slower)
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    return trainer


if __name__ == '__main__':
    # Test Lightning module
    print("Testing PyTorch Lightning Module")
    print("=" * 80)

    # Create dummy data
    batch_size = 32
    seq_len = 10
    input_dim = 4
    n_train = 1000
    n_val = 200

    X_train = np.random.randn(n_train, seq_len, input_dim)
    y_train = np.random.randn(n_train, input_dim)
    X_val = np.random.randn(n_val, seq_len, input_dim)
    y_val = np.random.randn(n_val, input_dim)
    X_test = np.random.randn(n_val, seq_len, input_dim)
    y_test = np.random.randn(n_val, input_dim)

    # Create DataModule
    datamodule = SeamGatedDataModule(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=batch_size,
        num_workers=0  # Use 0 for testing
    )

    # Create model
    model = SeamGatedLightningModule(
        model_type='z2_seam',
        input_dim=input_dim,
        hidden_dim=64,
        learning_rate=1e-3,
        max_epochs=5
    )

    print(f"Model type: {model.hparams.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = create_trainer(
        max_epochs=5,
        accelerator='cpu',  # Use CPU for testing
        devices=1,
        precision='32',
        enable_checkpointing=False,
        enable_early_stopping=False
    )

    print(f"\nTrainer configured:")
    print(f"  Max epochs: {trainer.max_epochs}")
    print(f"  Precision: {trainer.precision}")
    print(f"  Accelerator: {trainer.accelerator}")

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    print("\n✓ Lightning module test passed!")
