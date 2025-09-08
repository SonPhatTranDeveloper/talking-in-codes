import logging
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingCallback(CallbackAny2Vec):
    """Callback to track training progress."""

    def __init__(self) -> None:
        """Initialize the training callback."""
        self.epoch = 0
        self.losses = []
        self.previous_cumulative_loss = 0.0

    def on_epoch_begin(self, model: Word2Vec) -> None:
        """Log progress at the beginning of each epoch.

        Args:
            model: The Word2Vec model being trained
        """
        logger.info(f"Epoch {self.epoch + 1:3d} starting...")

    def on_epoch_end(self, model: Word2Vec) -> None:
        """Log progress at the end of each epoch.

        Args:
            model: The Word2Vec model being trained
        """
        # Get cumulative loss since training started
        cumulative_loss = model.get_latest_training_loss()

        # Calculate epoch loss as difference from previous cumulative loss
        if self.epoch == 0:
            # For first epoch, the cumulative loss is the epoch loss
            epoch_loss = cumulative_loss
        else:
            # For subsequent epochs, subtract previous cumulative loss
            epoch_loss = cumulative_loss - self.previous_cumulative_loss

        # Store losses for tracking
        self.losses.append(epoch_loss)

        # Log progress with better formatting
        logger.info(
            f"====    Epoch {self.epoch + 1:3d} | Loss: {epoch_loss:12.0f} | Cumulative: {cumulative_loss:12.0f}"
        )

        # Update tracking variables
        self.previous_cumulative_loss = cumulative_loss
        self.epoch += 1

    def get_loss_statistics(self) -> dict[str, float]:
        """Get training loss statistics.

        Returns:
            Dictionary containing loss statistics
        """
        if not self.losses:
            return {"message": "No loss data available"}

        valid_losses = [loss for loss in self.losses if loss > 0]

        if not valid_losses:
            return {
                "message": "No valid loss values recorded",
                "total_epochs": len(self.losses),
                "zero_loss_epochs": len(self.losses),
            }

        return {
            "total_epochs": len(self.losses),
            "valid_epochs": len(valid_losses),
            "first_epoch_loss": valid_losses[0] if valid_losses else 0.0,
            "last_epoch_loss": valid_losses[-1] if valid_losses else 0.0,
            "min_loss": min(valid_losses),
            "max_loss": max(valid_losses),
            "avg_loss": sum(valid_losses) / len(valid_losses),
            "total_loss_reduction": (valid_losses[0] - valid_losses[-1])
            if len(valid_losses) > 1
            else 0.0,
        }


class Word2VecEmbeddingsTrainer:
    """A comprehensive Word2Vec embedding trainer using Gensim.

    This class provides functionality to train, save, load, and use Word2Vec embeddings
    for semantic similarity tasks. It supports both CBOW and Skip-gram architectures
    with configurable hyperparameters. Expects pre-tokenized sentences as input.
    """

    def __init__(
        self,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 0,  # 0 for CBOW, 1 for Skip-gram
        hs: int = 0,  # 0 for negative sampling, 1 for hierarchical softmax
        negative: int = 5,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        seed: int = 42,
        compute_loss: bool = True,
        callbacks: list[CallbackAny2Vec] | None = None,
    ) -> None:
        """Initialize the Word2Vec trainer with configuration parameters.

        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum frequency count for words to be included in vocabulary
            workers: Number of worker threads for training
            sg: Training algorithm (0=CBOW, 1=Skip-gram)
            hs: Use hierarchical softmax (1) or negative sampling (0)
            negative: Number of negative samples (ignored if hs=1)
            alpha: Initial learning rate
            min_alpha: Final learning rate
            seed: Random seed for reproducibility
            compute_loss: Whether to compute and track training loss
            callbacks: Optional list of training callbacks
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        self.compute_loss = compute_loss
        self.callbacks = callbacks
        self.model: Word2Vec | None = None
        self.is_trained = False

        algorithm = "Skip-gram" if sg else "CBOW"
        logger.info("=" * 60)
        logger.info("Word2Vec Embedding Trainer Initialized")
        logger.info("=" * 60)
        logger.info(f"  Algorithm      : {algorithm}")
        logger.info(f"  Vector Size    : {vector_size}")
        logger.info(f"  Context Window : {window}")
        logger.info(f"  Min Count      : {min_count}")
        logger.info(f"  Workers        : {workers}")
        logger.info(f"  Learning Rate  : {alpha} -> {min_alpha}")
        logger.info(f"  Seed           : {seed}")
        logger.info(f"  Compute Loss   : {compute_loss}")
        logger.info("=" * 60)

    def train(
        self,
        sentences: list[list[str]],
        epochs: int = 10,
    ) -> None:
        """Train the Word2Vec model on the provided tokenized sentences.

        Args:
            sentences: List of tokenized sentences (each sentence is a list of words)
            epochs: Number of training epochs
            compute_loss: Whether to compute and track training loss
            callbacks: Optional list of training callbacks
        """
        if not sentences:
            raise ValueError("Cannot train on empty sentences list")

        # Validate sentences
        valid_sentences = [s for s in sentences if s]  # Filter out empty sentences
        if not valid_sentences:
            raise ValueError("No valid sentences after filtering empty ones")

        # Log training start
        logger.info("-" * 60)
        logger.info("Starting Word2Vec Training")
        logger.info("-" * 60)
        logger.info(f"  Input Sentences     : {len(sentences):,}")
        logger.info(f"  Valid Sentences     : {len(valid_sentences):,}")
        logger.info(f"  Training Epochs     : {epochs}")
        logger.info(f"  Compute Loss        : {self.compute_loss}")

        # Warn if loss computation is disabled but callbacks expect it
        if not self.compute_loss and self.callbacks:
            logger.warning(
                "  WARNING: Loss computation is disabled but callbacks are provided. "
                "Loss values will be 0 or unavailable."
            )

        logger.info("-" * 60)

        # Initialize model
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            seed=self.seed,
            compute_loss=self.compute_loss,
        )

        # Build vocabulary
        logger.info("PHASE 1: Building vocabulary...")
        self.model.build_vocab(valid_sentences)
        vocab_size = len(self.model.wv.key_to_index)

        if vocab_size == 0:
            total_words = sum(len(sentence) for sentence in valid_sentences)
            unique_words = len(
                {word for sentence in valid_sentences for word in sentence}
            )
            raise ValueError(
                f"No vocabulary was built! This usually happens when min_count={self.min_count} "
                f"is too high for your data. Found {unique_words} unique words from {total_words} "
                f"total words. Consider lowering min_count or providing more training data."
            )

        logger.info(f"  Vocabulary built successfully: {vocab_size:,} unique words")

        # Train model
        logger.info("PHASE 2: Training embeddings...")
        logger.info("  Training progress:")
        self.model.train(
            valid_sentences,
            total_examples=len(valid_sentences),
            epochs=epochs,
            callbacks=self.callbacks,
            compute_loss=self.compute_loss,
        )

        self.is_trained = True
        logger.info("-" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"  Final vocabulary size : {vocab_size:,} words")
        logger.info(f"  Vector dimensions     : {self.vector_size}")
        logger.info("-" * 60)

        print(self.model.get_latest_training_loss())

    def save_model(self, model_path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            model_path: Path where to save the model

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving Word2Vec model...")
        logger.info(f"  Destination: {model_path}")
        logger.info(
            f"  Model size:  {self.get_vocab_size():,} words x {self.vector_size} dimensions"
        )
        self.model.save(str(model_path))
        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"  File size:   {file_size:.1f} MB")
        logger.info("Model saved successfully!")

    def load_model(self, model_path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info("Loading Word2Vec model...")
        logger.info(f"  Source:    {model_path}")
        logger.info(f"  File size: {file_size:.1f} MB")

        self.model = Word2Vec.load(str(model_path))
        self.is_trained = True

        logger.info(
            f"  Loaded:    {self.get_vocab_size():,} words x {self.model.vector_size} dimensions"
        )
        logger.info("Model loaded successfully!")

    def get_word_vector(self, word: str) -> np.ndarray | None:
        """Get the vector representation of a word.

        Args:
            word: The word to get vector for

        Returns:
            Word vector as numpy array, or None if word not in vocabulary
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained or loaded before getting vectors")

        try:
            return self.model.wv[word]
        except KeyError:
            vocab_size = len(self.model.wv.key_to_index)
            logger.warning(
                f"Word '{word}' not found in vocabulary of {vocab_size:,} words"
            )
            return None

    def get_vocab_size(self) -> int:
        """Get the size of the model's vocabulary.

        Returns:
            Number of words in vocabulary
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "Model must be trained or loaded before getting vocab size"
            )

        return len(self.model.wv.key_to_index)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model.

        Returns:
            Dictionary containing model configuration and statistics
        """
        if not self.is_trained or self.model is None:
            raise ValueError(
                "Model must be trained or loaded before getting model info"
            )

        return {
            "vector_size": self.model.vector_size,
            "window": self.model.window,
            "min_count": self.model.min_count,
            "workers": self.model.workers,
            "sg": self.model.sg,
            "hs": self.model.hs,
            "negative": self.model.negative,
            "vocab_size": self.get_vocab_size(),
            "total_train_time": getattr(self.model, "total_train_time", "Unknown"),
        }
