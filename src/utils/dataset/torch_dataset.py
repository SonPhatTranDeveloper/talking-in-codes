import csv
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for loading word embeddings from a CSV file.

    The CSV must contain the column: "embedding". Only the "embedding" column is
    used for samples returned by this dataset.

    The "embedding" field should be a JSON array (e.g., "[0.1, 0.2, 0.3]"). If it's
    not valid JSON, a permissive fallback attempts to parse comma- or whitespace-
    separated numbers.

    Args:
        csv_path: Path to the CSV file.
        embedding_column: Name of the column containing the embedding vector.
        delimiter: CSV delimiter character.
        encoding: File encoding.

    Returns:
        Each dataset item is a numpy.ndarray of shape (vector_size,) and dtype float32.
    """

    def __init__(
        self,
        csv_path: str | Path,
        embedding_column: str = "embedding",
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.embedding_column = embedding_column
        self.delimiter = delimiter
        self.encoding = encoding

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self._embeddings: np.ndarray | None = None
        self._vector_size: int | None = None

        self._load()

    @property
    def vector_size(self) -> int:
        """Return the dimensionality of the embedding vectors.

        Returns:
            Embedding vector dimensionality as an integer.
        """
        if self._vector_size is None:
            raise RuntimeError("Dataset not initialized correctly: vector size unknown")
        return self._vector_size

    def __len__(self) -> int:
        """Return the number of embedding rows available.

        Returns:
            Number of samples in the dataset.
        """
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the embedding vector at the specified index as a numpy array.

        Args:
            index: Row index within the dataset.

        Returns:
            A numpy array of shape (vector_size,) and dtype float32.
        """
        if self._embeddings is None:
            raise IndexError("Dataset is empty; no embeddings loaded")
        return self._embeddings[index]

    def _load(self) -> None:
        """Load and parse the CSV file into a contiguous numpy array of embeddings.

        Raises:
            ValueError: If required columns are missing or rows have inconsistent sizes.
        """
        logger.info("Loading embeddings from CSV: %s", self.csv_path)

        rows: list[np.ndarray] = []

        with self.csv_path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)

            # Validate headers
            fieldnames = reader.fieldnames or []
            required = {self.embedding_column}
            missing = [c for c in required if c not in fieldnames]
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {missing}. Found columns: {fieldnames}"
                )

            for row_index, row in enumerate(reader):
                raw_embedding = row.get(self.embedding_column, "")
                if raw_embedding is None or raw_embedding == "":
                    raise ValueError(
                        f"Row {row_index} has empty '{self.embedding_column}' field"
                    )

                vector = self._parse_embedding(raw_embedding)

                if self._vector_size is None:
                    self._vector_size = int(vector.shape[0])
                elif vector.shape[0] != self._vector_size:
                    raise ValueError(
                        f"Row {row_index} has inconsistent embedding size: "
                        f"{vector.shape[0]} != {self._vector_size}"
                    )

                rows.append(vector.astype(np.float32, copy=False))

        if not rows:
            logger.warning("No rows were read from CSV: %s", self.csv_path)
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self._vector_size = 0
            return

        # Stack into a 2D array for efficient slicing
        self._embeddings = np.vstack(rows)
        logger.info(
            "Loaded %d embeddings of dimension %d", len(rows), self._vector_size
        )

    @staticmethod
    def _parse_embedding(value: str) -> np.ndarray:
        """Parse a string containing an embedding vector into a numpy array.

        This first tries JSON parsing (expected format: "[v1, v2, ...]"). If that
        fails, it falls back to splitting on commas or whitespace.

        Args:
            value: String representation of the embedding vector.

        Returns:
            A numpy array of dtype float32.
        """
        value_stripped = value.strip()

        # Try JSON array first
        if value_stripped.startswith("[") and value_stripped.endswith("]"):
            try:
                data = json.loads(value_stripped)
                return np.asarray(data, dtype=np.float32)
            except json.JSONDecodeError:
                # Fall through to permissive parsing
                pass

        # Permissive fallback: split by comma or whitespace
        # Replace commas with spaces then split
        tokens: Iterable[str] = value_stripped.replace(",", " ").split()
        try:
            return np.asarray([float(t) for t in tokens], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(
                f"Failed to parse embedding value: {value_stripped!r}"
            ) from exc
