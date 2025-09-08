import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.model.autoencoder import RQVAEAutoencoder
from src.utils.trainer.rq_vae_trainer import RQVAETrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataloader(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Load the dataloader from the config.

    Args:
        config: The config.

    Returns:
        The dataloader.
    """
    # Load the dataset
    dataset = hydra.utils.instantiate(config.dataset)

    # Split into train and validation
    train_length = int(len(dataset) * (1 - config.training.val_split))
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_length, val_length],
        generator=torch.Generator().manual_seed(config.training.seed),
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=config.training.persistent_workers,
    )

    # Log the dataloaders
    logger.info("=" * 80)
    logger.info("DATASET")
    logger.info(f"-- Train dataloader size: {len(train_dataset)} words")
    logger.info(f"-- Val dataloader size: {len(val_dataset)} words")
    logger.info("=" * 80)

    return train_dataloader, val_dataloader


def load_model(config: DictConfig) -> RQVAEAutoencoder:
    """
    Load the model from the config.

    Args:
        config: The config.

    Returns:
        The model.
    """
    model = hydra.utils.instantiate(config.model)

    # Log the model
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("=" * 80)
    logger.info("MODEL")
    logger.info(f"-- Number of model parameters: {num_params}")
    logger.info("=" * 80)

    return model


def load_optimizer(config: DictConfig, model: RQVAEAutoencoder) -> Optimizer:
    """
    Load the optimizer from the config.

    Args:
        config: The config.

    Returns:
        The optimizer.
    """
    # Load the optimizer
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    # Log the optimizer
    logger.info("=" * 80)
    logger.info("OPTIMIZER")
    logger.info(f"-- Learning rate: {config.optimizer.lr}")
    logger.info("=" * 80)

    return optimizer


@hydra.main(
    config_path="../../src/config", config_name="config.yaml", version_base=None
)
def main(config: DictConfig) -> None:
    """
    Main function to train the autoencoder.
    """
    # Load the dataloader
    train_dataloader, val_dataloader = load_dataloader(config)

    # Load the model
    model = load_model(config)

    # Load the optimizer
    optimizer = load_optimizer(config, model)

    # Load the device
    device = torch.device(config.training.device)

    # Load the trainer
    trainer = RQVAETrainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        alpha=config.training.alpha,
    )

    # Train the model
    trainer.train(config.training.epochs)

    # Save the model
    trainer.save_model(config.training.save_model_path)


if __name__ == "__main__":
    main()
