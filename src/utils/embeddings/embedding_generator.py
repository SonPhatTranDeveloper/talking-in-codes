import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

logging.basicConfig(level=logging.INFO)


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def get_embedding(self, word: str) -> np.ndarray:
        raise NotImplementedError


class Word2VecEmbeddingGenerator(EmbeddingGenerator):
    """
    Generator for Word2Vec embeddings.

    Load two models:
        - Pre-trained model from gensim
        - Custom trained model from our own training script

    Interpolate between the two models using the interpolation factor.
    """

    def __init__(
        self,
        model_path: str,
    ):
        """Initialize the Word2Vec embedding generator.

        Args:
            model_path: Filesystem path to the custom trained Word2Vec model saved by our trainer.
            interpolation_factor: Weight given to the custom model during interpolation in [0, 1].

        Raises:
            ValueError: If interpolation_factor is not within [0, 1].
            FileNotFoundError: If the custom model path does not exist.
        """
        self._logger = logging.getLogger(__name__)

        # Load custom trained Word2Vec model and access its KeyedVectors
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Custom model file not found: {model_path_obj}")

        self._logger.info("Loading custom Word2Vec model from: %s", model_path_obj)
        custom_model: Word2Vec = Word2Vec.load(str(model_path_obj))
        self.custom_kv: KeyedVectors = custom_model.wv
        self._logger.info(
            "Loaded custom model with vector size: %d and vocab size: %d",
            self.custom_kv.vector_size,
            len(self.custom_kv.key_to_index),
        )

    def get_embedding(self, word: str) -> np.ndarray:
        """Get the embedding for a word using interpolation between two sources.

        If the word exists in pretrained, prioritize it over the custom model.
        Else if the word exists in custom, return its vector.
        Else raise an error.

        Args:
            word: The input word to embed.

        Returns:
            The embedding vector as a numpy array.

        Raises:
            KeyError: If the word is not present in either vocabulary.
        """
        in_custom = word in self.custom_kv.key_to_index

        if in_custom:
            return self.custom_kv[word]

        self._logger.warning("Word '%s' not found in either vocabulary", word)
        raise KeyError(f"Word not found in either vocabulary: {word}")


if __name__ == "__main__":
    model = Word2Vec.load("models/word2vec/wiki_skipgram_v1.model")
    print(model.wv.most_similar("tuesday"))
