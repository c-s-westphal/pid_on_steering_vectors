"""
Linear probe trainer for concept classification.
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


class LinearProbe(nn.Module):
    """Simple linear classifier for concept detection."""

    def __init__(self, input_dim: int, num_classes: int = 4):
        """
        Initialize linear probe.

        Args:
            input_dim: Dimension of input activations (hidden_size)
            num_classes: Number of classes (4: neither/dog/bridge/both)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)

    def get_steering_vectors(self) -> torch.Tensor:
        """
        Extract steering vectors from probe weights.

        Returns:
            Tensor of shape (num_classes, input_dim)
            Each row is a steering vector for that class
        """
        return self.linear.weight.data.clone()


class ProbeTrainer:
    """Trainer for linear probe."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize trainer.

        Args:
            input_dim: Dimension of activations
            num_classes: Number of classes
            learning_rate: Learning rate for optimizer
            device: Device to train on
            dtype: Data type for model (should match activation dtype)
        """
        self.device = device
        self.dtype = dtype
        self.model = LinearProbe(input_dim, num_classes).to(device).to(dtype)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(
        self,
        train_activations: torch.Tensor,
        train_labels: torch.Tensor,
        val_activations: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_epochs: int = 50,
        batch_size: int = 32
    ) -> dict:
        """
        Train the linear probe.

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
        # Create datasets
        train_dataset = ActivationDataset(train_activations, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_activations is not None:
            val_dataset = ActivationDataset(val_activations, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        logger.info(f"Training probe for {num_epochs} epochs...")

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
                loss.backward()
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

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")

        logger.info("Training complete!")
        return history

    def get_steering_vectors(self) -> torch.Tensor:
        """
        Extract steering vectors from trained probe.

        Returns:
            Tensor of shape (num_classes, hidden_size)
        """
        return self.model.get_steering_vectors()

    def save(self, path: str):
        """Save trained probe."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.linear.in_features,
            'num_classes': self.model.linear.out_features
        }, path)
        logger.info(f"Saved probe to {path}")

    def load(self, path: str):
        """Load trained probe."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded probe from {path}")


if __name__ == "__main__":
    # Test the probe trainer
    logging.basicConfig(level=logging.INFO)

    # Dummy data
    hidden_size = 3584  # Qwen 7B
    num_samples = 400
    num_classes = 4

    train_activations = torch.randn(num_samples, hidden_size)
    train_labels = torch.randint(0, num_classes, (num_samples,))

    trainer = ProbeTrainer(input_dim=hidden_size, num_classes=num_classes)
    history = trainer.train(train_activations, train_labels, num_epochs=20)

    steering_vectors = trainer.get_steering_vectors()
    print(f"\nSteering vectors shape: {steering_vectors.shape}")
    print(f"Row 0 (neither): norm = {torch.norm(steering_vectors[0]).item():.4f}")
    print(f"Row 1 (dog): norm = {torch.norm(steering_vectors[1]).item():.4f}")
    print(f"Row 2 (bridge): norm = {torch.norm(steering_vectors[2]).item():.4f}")
    print(f"Row 3 (both): norm = {torch.norm(steering_vectors[3]).item():.4f}")
