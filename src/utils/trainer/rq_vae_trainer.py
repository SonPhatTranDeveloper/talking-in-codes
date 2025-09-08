import logging
from typing import Any

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils.model.autoencoder import RQVAEAutoencoder


class RQVAETrainer:
    """
    Trainer for RQVA model.

    RQVA model is a model that uses a residual vector quantizer to compress the latent
    representation of the input.

    Args:
        model: The RQVA model to train.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        optimizer: The optimizer to use for training.
    """

    def __init__(
        self,
        model: RQVAEAutoencoder,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
        alpha: float = 1.0,
    ) -> None:
        """
        Initialize the RQVA trainer.

        Args:
            model: The RQVA model to train.
            train_dataloader: The training dataloader.
            val_dataloader: The validation dataloader.
            optimizer: The optimizer to use for training.
            device: Torch device to run training on.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.alpha = alpha

    def _prepare_inputs(self, batch: Any) -> torch.Tensor:
        """
        Convert a batch from the dataloader into a float32 tensor on the target device.

        Args:
            batch: Batch item from the dataloader, typically a tensor or numpy array.

        Returns:
            Tensor of shape (batch_size, feature_dim) on the specified device.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(batch, dtype=torch.float32, device=self.device)

    def _compute_losses(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the model forward pass and compute total, reconstruction, and commitment losses.

        Args:
            inputs: Input tensor.
            mse_loss: Loss function for reconstruction (MSE).

        Returns:
            Tuple of (total_loss, recon_loss, commitment_loss_mean).
        """
        quantized, indices, commitment_loss = self.model(inputs)

        commitment_loss_mean = commitment_loss.mean()

        quantized = quantized.clamp(-1, 1)
        recon_loss = (quantized - inputs).abs().mean()
        total_loss = recon_loss + self.alpha * commitment_loss_mean
        return total_loss, recon_loss, commitment_loss_mean

    def _run_epoch(
        self,
        dataloader: DataLoader,
        training: bool,
        description: str,
    ) -> tuple[float, float, float]:
        """
        Execute a single training or validation epoch.

        Args:
            dataloader: DataLoader to iterate over.
            mse_loss: MSE loss instance for reconstruction.
            training: If True, run in training mode with optimization; else eval mode.
            description: Text to display in the tqdm progress bar.

        Returns:
            Averages tuple: (avg_total_loss, avg_recon_loss, avg_commitment_loss).
        """
        self.model.train(mode=training)
        total_sum = 0.0
        recon_sum = 0.0
        commit_sum = 0.0
        batch_count = 0

        with torch.set_grad_enabled(training):
            progress = tqdm(dataloader, desc=description, leave=False)
            for batch in progress:
                inputs = self._prepare_inputs(batch)
                total_loss, recon_loss, commit_loss = self._compute_losses(inputs)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    self.optimizer.step()

                total_sum += float(total_loss.detach().cpu())
                recon_sum += float(recon_loss.detach().cpu())
                commit_sum += float(commit_loss.detach().cpu())
                batch_count += 1

                denom_live = max(1, batch_count)
                progress.set_postfix(
                    loss=f"{total_sum / denom_live:.4f}",
                    recon=f"{recon_sum / denom_live:.4f}",
                    commit=f"{commit_sum / denom_live:.4f}",
                )

        denom = max(1, batch_count)
        return total_sum / denom, recon_sum / denom, commit_sum / denom

    def train(self, epochs: int) -> None:
        """
        Train the RQVA model.

        Args:
            epochs: Number of epochs to train for.

        Returns:
            None
        """
        for epoch in range(1, int(epochs) + 1):
            avg_train_loss, avg_train_recon, avg_train_commit = self._run_epoch(
                self.train_dataloader,
                training=True,
                description=f"Train {epoch}/{epochs}",
            )

            avg_val_loss, avg_val_recon, avg_val_commit = self._run_epoch(
                self.val_dataloader,
                training=False,
                description=f"Val {epoch}/{epochs}",
            )

            self._logger.info(
                "Epoch %d/%d - train_loss: %.6f (recon %.6f, commit %.6f) - val_loss: %.6f (recon %.6f, commit %.6f)",
                epoch,
                epochs,
                avg_train_loss,
                avg_train_recon,
                avg_train_commit,
                avg_val_loss,
                avg_val_recon,
                avg_val_commit,
            )

    def save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: The path to save the model to.
        """
        torch.save(self.model.state_dict(), path)
