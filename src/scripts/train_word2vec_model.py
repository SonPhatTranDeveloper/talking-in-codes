#!/usr/bin/env python3
"""
Word2Vec Model Training Script

This script trains a Word2Vec model using WikiText datasets and saves the trained model.
It uses the WikiLoader for data loading and Word2VecEmbeddingsTrainer for model training.

Example usage:

    # Train with default parameters (CBOW, wikitext-2-v1, 300 dimensions, 10 epochs)
    uv run src/scripts/train_word2vec_model.py

    # Train Skip-gram model with custom vector size and epochs
    uv run src/scripts/train_word2vec_model.py --sg 1 --vector-size 100 --epochs 20

    # Use specific dataset splits
    uv run src/scripts/train_word2vec_model.py --splits train validation

    # Train on larger dataset with custom output name
    uv run src/scripts/train_word2vec_model.py --dataset-name wikitext-103-v1 --model-name large_wiki_model

    # Custom training configuration for production model
    uv run src/scripts/train_word2vec_model.py \
        --dataset-name wikitext-103-v1 \
        --splits train validation test \
        --sg 1 \
        --window 10 \
        --min-count 5 \
        --epochs 10 \
        --workers 12 \
        --output-dir models/word2vec \
        --compute-loss \
        --model-name wiki_skipgram_v1

    # Quick test run with minimal parameters
    uv run src/scripts/train_word2vec_model.py --vector-size 50 --epochs 3 --min-count 1
"""

import argparse
import logging
from pathlib import Path

from src.utils.dataset.wiki_loader import LowercaseWordProcessor, WikiLoader
from src.utils.embeddings.word2vec_trainer import (
    TrainingCallback,
    Word2VecEmbeddingsTrainer,
)
from src.utils.tokenizer.word_tokenizer import NLTKWordTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model using WikiText dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext-2-v1",
        help="Name of the WikiText dataset to use",
    )
    dataset_group.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to use for training",
    )
    # Tokenizer is always NLTK by default; no CLI flag needed

    # Word2Vec model arguments
    model_group = parser.add_argument_group("Word2Vec Model Configuration")
    model_group.add_argument(
        "--vector-size", type=int, default=300, help="Dimensionality of word vectors"
    )
    model_group.add_argument(
        "--window", type=int, default=5, help="Context window size"
    )
    model_group.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum frequency count for words to be included in vocabulary",
    )
    model_group.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads for training"
    )
    model_group.add_argument(
        "--sg",
        type=int,
        choices=[0, 1],
        default=0,
        help="Training algorithm: 0 for CBOW, 1 for Skip-gram",
    )
    model_group.add_argument(
        "--hs",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use hierarchical softmax (1) or negative sampling (0)",
    )
    model_group.add_argument(
        "--negative",
        type=int,
        default=5,
        help="Number of negative samples (ignored if hs=1)",
    )
    model_group.add_argument(
        "--alpha", type=float, default=0.025, help="Initial learning rate"
    )
    model_group.add_argument(
        "--min-alpha", type=float, default=0.0001, help="Final learning rate"
    )
    model_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Training arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    training_group.add_argument(
        "--compute-loss",
        action="store_true",
        default=True,
        help="Whether to compute and track training loss",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the trained model",
    )
    output_group.add_argument(
        "--model-name",
        type=str,
        help="Name for the saved model file (auto-generated if not provided)",
    )

    return parser.parse_args()


def generate_model_name(args: argparse.Namespace) -> str:
    """Generate a descriptive model name based on training parameters.

    Args:
        args: Parsed command line arguments

    Returns:
        Generated model filename
    """
    algorithm = "skipgram" if args.sg else "cbow"
    dataset_name = args.dataset_name.replace("-", "_")

    model_name = (
        f"word2vec_{algorithm}_{dataset_name}_"
        f"d{args.vector_size}_w{args.window}_mc{args.min_count}_"
        f"e{args.epochs}.model"
    )
    return model_name


def main() -> None:
    """Main training function."""
    args = parse_arguments()

    logger.info("Starting Word2Vec training script")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Splits: {', '.join(args.splits)}")
    logger.info(f"  Algorithm: {'Skip-gram' if args.sg else 'CBOW'}")
    logger.info(f"  Vector size: {args.vector_size}")
    logger.info(f"  Window size: {args.window}")
    logger.info(f"  Min count: {args.min_count}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info("=" * 80)

    # Initialize WikiLoader
    logger.info("Initializing WikiLoader...")
    wiki_loader = WikiLoader(
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        word_tokenizer=NLTKWordTokenizer(),
        word_processor=LowercaseWordProcessor(),
    )

    # Load training data
    logger.info("Loading training data...")
    sentences = wiki_loader.load_data()

    if not sentences:
        logger.error("No training data loaded. Exiting.")
        return

    logger.info(f"Successfully loaded {len(sentences):,} sentences")

    # Initialize Word2Vec trainer with callback
    logger.info("Initializing Word2Vec trainer...")
    training_callback = TrainingCallback()
    trainer = Word2VecEmbeddingsTrainer(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        hs=args.hs,
        negative=args.negative,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        seed=args.seed,
        compute_loss=args.compute_loss,
        callbacks=[training_callback],
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.train(
        sentences=sentences,
        epochs=args.epochs,
    )

    # Generate model name if not provided
    if args.model_name is None:
        model_name = generate_model_name(args)
    else:
        model_name = args.model_name
        if not model_name.endswith(".model"):
            model_name += ".model"

    # Save the trained model
    model_path = args.output_dir / model_name
    logger.info("Saving trained model...")
    trainer.save_model(model_path)

    # Display model information and loss statistics
    model_info = trainer.get_model_info()
    loss_stats = training_callback.get_loss_statistics()

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("-" * 80)
    logger.info("Training Loss Statistics:")
    if "message" in loss_stats:
        logger.info(f"  {loss_stats['message']}")
        if "zero_loss_epochs" in loss_stats:
            logger.info(
                f"  Total epochs with zero loss: {loss_stats['zero_loss_epochs']}/{loss_stats['total_epochs']}"
            )
            logger.info(
                "  NOTE: Zero loss may indicate compute_loss=False or loss computation issues"
            )
    else:
        for key, value in loss_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:,.0f}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
