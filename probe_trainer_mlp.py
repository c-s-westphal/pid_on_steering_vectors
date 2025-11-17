"""
MLP probe trainer for concept classification with better stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ActivationDataset(Dataset):
    """Dataset of activations and labels."""

    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset.

        Args:
            activations: Tensor of shape (num_samples, hidden_size)
            labels: Tensor of shape (num_samples,) with class labels
        """
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


class MLPProbe(nn.Module):
    """MLP classifier for concept detection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 4):
        """
        Initialize MLP probe.

        Args:
            input_dim: Dimension of input activations (hidden_size)
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes (4: neither/dog/bridge/both)
        """
        super().__init__()
        # Split into layers so we can access bottleneck
        # No normalization to preserve L2 norms
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Forward pass."""
        # Pass through input layer (bottleneck)
        bottleneck = self.input_layer(x)

        # Continue through rest of network
        x = self.activation(bottleneck)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def get_final_layer_weights(self) -> torch.Tensor:
        """
        Extract final layer weights as steering vectors.

        Returns:
            Tensor of shape (num_classes, hidden_dim)
            Each row is the weight vector for that class
        """
        return self.output_layer.weight.data.clone()

    def get_bottleneck_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get bottleneck activation for given input.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Bottleneck activation of shape (batch_size, hidden_dim) or (hidden_dim,)
        """
        with torch.no_grad():
            return self.input_layer(x)


class MLPProbeTrainer:
    """Trainer for MLP probe with better numerical stability."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_classes: int = 4,
        learning_rate: float = 1e-4,  # Lower LR for stability
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_float32: bool = True  # Force float32 for stability
    ):
        """
        Initialize trainer.

        Args:
            input_dim: Dimension of activations
            hidden_dim: Hidden layer dimension (defaults to input_dim)
            num_classes: Number of classes
            learning_rate: Learning rate for optimizer
            device: Device to train on
            use_float32: Use float32 instead of float16 for numerical stability
        """
        self.device = device
        self.use_float32 = use_float32

        if hidden_dim is None:
            hidden_dim = input_dim  # Same as model width

        self.model = MLPProbe(input_dim, hidden_dim, num_classes).to(device)

        if use_float32:
            self.model = self.model.float()
            logger.info("Using float32 for probe (better stability)")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def train(
        self,
        train_activations: torch.Tensor,
        train_labels: torch.Tensor,
        val_activations: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        batch_size: int = 32
    ) -> dict:
        """
        Train the MLP probe.

        Args:
            train_activations: Training activations (num_samples, hidden_size)
            train_labels: Training labels (num_samples,)
            val_activations: Validation activations (optional)
            val_labels: Validation labels (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Dictionary with training history
        """
        # Convert to float32 if needed
        if self.use_float32:
            train_activations = train_activations.float()
            if val_activations is not None:
                val_activations = val_activations.float()

        # Create datasets
        train_dataset = ActivationDataset(train_activations, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_activations is not None:
            val_dataset = ActivationDataset(val_activations, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        logger.info(f"Training MLP probe for {num_epochs} epochs...")
        logger.info(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_activations, batch_labels in train_loader:
                batch_activations = batch_activations.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_activations)
                loss = self.criterion(outputs, batch_labels)

                # Check for NaN
                if torch.isnan(loss):
                    logger.error(f"NaN loss at epoch {epoch+1}!")
                    break

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validation
            if val_activations is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_activations, batch_labels in val_loader:
                        batch_activations = batch_activations.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        outputs = self.model(batch_activations)
                        loss = self.criterion(outputs, batch_labels)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_labels.size(0)
                        val_correct += predicted.eq(batch_labels).sum().item()

                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        return history

    def get_steering_vectors(self, method: str = 'final_layer') -> torch.Tensor:
        """
        Extract steering vectors from trained probe.

        Args:
            method: 'final_layer' for final layer weights (default for backwards compatibility)

        Returns:
            Tensor of shape (num_classes, hidden_dim)
        """
        if method == 'final_layer':
            return self.model.get_final_layer_weights()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'final_layer' or call get_bottleneck_steering_vectors() directly.")

    def get_bottleneck_steering_vectors(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Extract steering vectors from bottleneck activations.

        This computes the bottleneck layer output for each sample, then aggregates
        by class to get one steering vector per class.

        Args:
            activations: Input activations of shape (num_samples, input_dim)
            labels: Labels for each sample of shape (num_samples,)
            aggregation: How to aggregate activations per class ('mean' or 'median')

        Returns:
            Tensor of shape (num_classes, hidden_dim)
            Each row is the aggregated bottleneck activation for that class
        """
        self.model.eval()

        # Convert to float32 if needed
        if self.use_float32:
            activations = activations.float()

        # Get bottleneck activations for all samples
        with torch.no_grad():
            activations = activations.to(self.device)
            bottleneck_acts = self.model.get_bottleneck_activation(activations)

        # Get unique classes
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        hidden_dim = bottleneck_acts.shape[1]

        # Aggregate by class
        steering_vectors = torch.zeros(num_classes, hidden_dim, device=self.device)

        for i, label in enumerate(sorted(unique_labels.tolist())):
            mask = labels == label
            class_activations = bottleneck_acts[mask]

            if aggregation == 'mean':
                steering_vectors[i] = class_activations.mean(dim=0)
            elif aggregation == 'median':
                steering_vectors[i] = class_activations.median(dim=0).values
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

        logger.info(f"Extracted bottleneck steering vectors using {aggregation} aggregation")
        logger.info(f"  Shape: {steering_vectors.shape}")
        for i in range(num_classes):
            norm = torch.norm(steering_vectors[i]).item()
            logger.info(f"  Class {i}: L2 norm = {norm:.4f}")

        return steering_vectors

    def save(self, path: str):
        """Save trained probe."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_layer.in_features,
            'hidden_dim': self.model.input_layer.out_features,
            'num_classes': self.model.output_layer.out_features
        }, path)
        logger.info(f"Saved probe to {path}")

    def load(self, path: str):
        """Load trained probe."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded probe from {path}")


if __name__ == "__main__":
    # Test the MLP probe trainer
    logging.basicConfig(level=logging.INFO)

    # Dummy data
    hidden_size = 2048  # Qwen 3B
    num_samples = 400
    num_classes = 4

    train_activations = torch.randn(num_samples, hidden_size)
    train_labels = torch.randint(0, num_classes, (num_samples,))

    trainer = MLPProbeTrainer(
        input_dim=hidden_size,
        hidden_dim=hidden_size,
        num_classes=num_classes
    )

    history = trainer.train(train_activations, train_labels, num_epochs=20)

    steering_vectors = trainer.get_steering_vectors()
    print(f"\nSteering vectors shape: {steering_vectors.shape}")
    print(f"Row 0 (neither): norm = {torch.norm(steering_vectors[0]).item():.4f}")
    print(f"Row 1 (dog): norm = {torch.norm(steering_vectors[1]).item():.4f}")
    print(f"Row 2 (bridge): norm = {torch.norm(steering_vectors[2]).item():.4f}")
    print(f"Row 3 (both): norm = {torch.norm(steering_vectors[3]).item():.4f}")
