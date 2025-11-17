"""
Split-half probe training for steering vector extraction.

Train binary linear probes on different halves of activation dimensions:
- First half (neurons 0:N/2): Dog detection
- Second half (neurons N/2:N): Bridge detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class BinaryLinearProbe(nn.Module):
    """Simple linear probe for binary classification."""

    def __init__(self, input_dim: int):
        """
        Initialize binary linear probe.

        Args:
            input_dim: Dimension of input activations
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass returning logits."""
        return self.linear(x)

    def get_weights(self) -> torch.Tensor:
        """
        Get probe weights as steering vector.

        Returns:
            Tensor of shape (input_dim,)
        """
        return self.linear.weight.data.squeeze().clone()


class SplitHalfProbeTrainer:
    """Trainer for split-half binary probes."""

    def __init__(
        self,
        full_dim: int,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_float32: bool = True
    ):
        """
        Initialize trainer for split-half probes.

        Args:
            full_dim: Full activation dimension (will be split in half)
            learning_rate: Learning rate for optimizer
            device: Device to train on
            use_float32: Use float32 for numerical stability
        """
        self.full_dim = full_dim
        self.half_dim = full_dim // 2
        self.device = device
        self.use_float32 = use_float32

        # Create binary probes for each half
        self.dog_probe = BinaryLinearProbe(self.half_dim).to(device)
        self.bridge_probe = BinaryLinearProbe(self.half_dim).to(device)

        if use_float32:
            self.dog_probe = self.dog_probe.float()
            self.bridge_probe = self.bridge_probe.float()

        self.criterion = nn.BCEWithLogitsLoss()

        logger.info(f"Created split-half probes:")
        logger.info(f"  Dog probe: neurons [0:{self.half_dim}]")
        logger.info(f"  Bridge probe: neurons [{self.half_dim}:{full_dim}]")

    def _prepare_binary_labels(self, labels: torch.Tensor, concept: str) -> torch.Tensor:
        """
        Convert 4-class labels to binary labels for a specific concept.

        Args:
            labels: Tensor of shape (N,) with values 0=neither, 1=dog, 2=bridge, 3=both
            concept: 'dog' or 'bridge'

        Returns:
            Binary labels (N,) with 1 if concept present, 0 otherwise
        """
        if concept == 'dog':
            # Dog present in classes 1 (dog) and 3 (both)
            return ((labels == 1) | (labels == 3)).float()
        elif concept == 'bridge':
            # Bridge present in classes 2 (bridge) and 3 (both)
            return ((labels == 2) | (labels == 3)).float()
        else:
            raise ValueError(f"Unknown concept: {concept}")

    def train_binary_probe(
        self,
        probe: BinaryLinearProbe,
        activations: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32,
        concept_name: str = ""
    ) -> Dict:
        """
        Train a single binary probe.

        Args:
            probe: The probe to train
            activations: Input activations
            labels: Binary labels
            num_epochs: Number of epochs
            batch_size: Batch size
            concept_name: Name for logging

        Returns:
            Training history dict
        """
        if self.use_float32:
            activations = activations.float()
            labels = labels.float()

        # Create dataset
        dataset = torch.utils.data.TensorDataset(activations, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)

        history = {'train_loss': [], 'train_acc': []}

        logger.info(f"Training {concept_name} probe for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            probe.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_act, batch_labels in dataloader:
                batch_act = batch_act.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = probe(batch_act).squeeze()
                loss = self.criterion(logits, batch_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

            epoch_loss /= len(dataloader)
            epoch_acc = 100.0 * correct / total

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)

            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

        logger.info(f"  Final accuracy: {epoch_acc:.2f}%")
        return history

    def train(
        self,
        train_activations: torch.Tensor,
        train_labels: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32
    ) -> Tuple[Dict, Dict]:
        """
        Train both split-half probes.

        Args:
            train_activations: Full activations of shape (num_samples, full_dim)
            train_labels: Labels of shape (num_samples,) with values 0-3
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Tuple of (dog_history, bridge_history)
        """
        # Split activations by dimension
        dog_activations = train_activations[:, :self.half_dim]
        bridge_activations = train_activations[:, self.half_dim:]

        # Prepare binary labels
        dog_labels = self._prepare_binary_labels(train_labels, 'dog')
        bridge_labels = self._prepare_binary_labels(train_labels, 'bridge')

        logger.info(f"Dog examples: {dog_labels.sum().item()}/{len(dog_labels)} positive")
        logger.info(f"Bridge examples: {bridge_labels.sum().item()}/{len(bridge_labels)} positive")

        # Train dog probe on first half
        dog_history = self.train_binary_probe(
            self.dog_probe,
            dog_activations,
            dog_labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            concept_name="Dog"
        )

        # Train bridge probe on second half
        bridge_history = self.train_binary_probe(
            self.bridge_probe,
            bridge_activations,
            bridge_labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            concept_name="Bridge"
        )

        return dog_history, bridge_history

    def get_concatenated_vector(self) -> torch.Tensor:
        """
        Method 1: Get steering vector by concatenating probe weights.

        Returns:
            Tensor of shape (full_dim,) with dog weights in first half,
            bridge weights in second half
        """
        dog_weights = self.dog_probe.get_weights()
        bridge_weights = self.bridge_probe.get_weights()

        # Concatenate: [dog_weights, bridge_weights]
        concatenated = torch.cat([dog_weights, bridge_weights], dim=0)

        logger.info(f"Concatenated vector: {concatenated.shape}")
        logger.info(f"  Dog half norm: {torch.norm(dog_weights).item():.4f}")
        logger.info(f"  Bridge half norm: {torch.norm(bridge_weights).item():.4f}")
        logger.info(f"  Total norm: {torch.norm(concatenated).item():.4f}")

        return concatenated

    def get_dog_direction(self) -> torch.Tensor:
        """Get dog direction (for MLP-weighted combination)."""
        # Pad dog weights with zeros in bridge half
        dog_weights = self.dog_probe.get_weights()
        padded = torch.cat([dog_weights, torch.zeros(self.half_dim, device=self.device)])
        return padded

    def get_bridge_direction(self) -> torch.Tensor:
        """Get bridge direction (for MLP-weighted combination)."""
        # Pad bridge weights with zeros in dog half
        bridge_weights = self.bridge_probe.get_weights()
        padded = torch.cat([torch.zeros(self.half_dim, device=self.device), bridge_weights])
        return padded

    def save(self, path: Path):
        """Save both probes."""
        torch.save({
            'dog_probe_state': self.dog_probe.state_dict(),
            'bridge_probe_state': self.bridge_probe.state_dict(),
            'full_dim': self.full_dim,
            'half_dim': self.half_dim
        }, path)
        logger.info(f"Saved split-half probes to {path}")

    def load(self, path: Path):
        """Load both probes."""
        checkpoint = torch.load(path)
        self.dog_probe.load_state_dict(checkpoint['dog_probe_state'])
        self.bridge_probe.load_state_dict(checkpoint['bridge_probe_state'])
        logger.info(f"Loaded split-half probes from {path}")


if __name__ == "__main__":
    # Test the split-half probe trainer
    logging.basicConfig(level=logging.INFO)

    # Dummy data
    full_dim = 2048
    num_samples = 400

    # Create random activations and labels
    activations = torch.randn(num_samples, full_dim)
    labels = torch.randint(0, 4, (num_samples,))  # 0=neither, 1=dog, 2=bridge, 3=both

    # Train split-half probes
    trainer = SplitHalfProbeTrainer(full_dim=full_dim)
    dog_hist, bridge_hist = trainer.train(activations, labels, num_epochs=50)

    # Get concatenated steering vector (Method 1)
    concat_vector = trainer.get_concatenated_vector()
    print(f"\nConcatenated vector shape: {concat_vector.shape}")
    print(f"Concatenated vector norm: {torch.norm(concat_vector).item():.4f}")

    # Get individual directions (for Method 2)
    dog_dir = trainer.get_dog_direction()
    bridge_dir = trainer.get_bridge_direction()
    print(f"\nDog direction shape: {dog_dir.shape}")
    print(f"Bridge direction shape: {bridge_dir.shape}")
