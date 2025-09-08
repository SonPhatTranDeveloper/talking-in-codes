import logging
from abc import ABC, abstractmethod

from datasets import load_dataset

from src.utils.tokenizer.word_tokenizer import NLTKWordTokenizer, WordTokenizer

logger = logging.getLogger(__name__)


class WordProcessor(ABC):
    @abstractmethod
    def process(self, word: str) -> str:
        raise NotImplementedError


class LowercaseWordProcessor(WordProcessor):
    def process(self, word: str) -> str:
        return word.lower()


class WikiLoader:
    """Loader for WikiText datasets using HuggingFace datasets library."""

    def __init__(
        self,
        dataset_name: str,
        splits: tuple[str, ...] = ("train", "validation", "test"),
        word_tokenizer: WordTokenizer | None = None,
        word_processor: WordProcessor | None = None,
    ) -> None:
        """Initialize WikiLoader.

        Args:
            dataset_name: Name of the dataset to load (e.g., "wikitext-2-v1")
            splits: Tuple of dataset splits to load
            word_tokenizer: Optional tokenizer to process the text data
            word_processor: Optional processor to process the words
        """
        self.dataset_name = dataset_name
        self.splits = splits
        # Default to NLTK tokenizer if none provided
        self.word_tokenizer = word_tokenizer or NLTKWordTokenizer()
        self.word_processor = word_processor
        logger.info(f"Initialized WikiLoader for dataset: {dataset_name}")

    def load_data(self) -> list[list[str]]:
        """Load and process WikiText dataset.

        Returns:
            List of tokenized text samples, where each sample is a list of tokens
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset("wikitext", self.dataset_name)

        all_texts = []

        # Process each split
        for split in self.splits:
            if split in dataset:
                logger.info(f"Processing split: {split}")
                split_data = dataset[split]

                # Extract text from each sample
                for sample in split_data:
                    text = sample["text"].strip()

                    # Skip empty lines
                    if not text:
                        continue

                    # Tokenize
                    tokens = self.word_tokenizer.tokenize(text)

                    # Optional post-tokenization processing
                    if self.word_processor:
                        tokens = [self.word_processor.process(word) for word in tokens]

                    # Only add non-empty token lists
                    if tokens:
                        all_texts.append(tokens)
            else:
                logger.warning(f"Split '{split}' not found in dataset")

        logger.info(f"Loaded {len(all_texts)} text samples")
        return all_texts
