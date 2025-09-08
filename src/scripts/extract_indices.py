#!/usr/bin/env python3
"""Script to extract indices from embeddings using a trained RQ-VAE model.

This script loads embeddings from a CSV file (word, count, embedding), uses a trained
RQ-VAE model to extract quantized indices for each embedding, and saves the results
to an output CSV file (word, count, indices).

How to run:

uv run src/scripts/extract_indices.py \
    --input-csv data/embeddings/words.csv \
    --output-csv data/indices/word_indices.csv \
    --model-path models/rq_vae/rqvae_autoencoder.pth \
    --device cpu \
    --batch-size 128
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.model.autoencoder import RQVAEAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_embedding(value: str) -> np.ndarray:
    """Parse a string containing an embedding vector into a numpy array.

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
            pass

    # Fallback: split by comma or whitespace
    tokens = value_stripped.replace(",", " ").split()
    try:
        return np.asarray([float(t) for t in tokens], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse embedding value: {value_stripped!r}"
        ) from exc


def load_embeddings_csv(
    csv_path: Path,
    word_column: str = "word",
    count_column: str = "count",
    embedding_column: str = "embedding",
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> list[dict[str, Any]]:
    """Load embeddings from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        word_column: Name of the column containing the word.
        count_column: Name of the column containing the count.
        embedding_column: Name of the column containing the embedding vector.
        delimiter: CSV delimiter character.
        encoding: File encoding.

    Returns:
        List of dictionaries containing word, count, and embedding data.
    """
    logger.info("Loading embeddings from CSV: %s", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = []

    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        # Validate headers
        fieldnames = reader.fieldnames or []
        required = {word_column, count_column, embedding_column}
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. Found columns: {fieldnames}"
            )

        for row_index, row in enumerate(reader):
            word = row.get(word_column, "").strip()
            count_str = row.get(count_column, "").strip()
            embedding_str = row.get(embedding_column, "").strip()

            if not word:
                raise ValueError(f"Row {row_index} has empty '{word_column}' field")
            if not count_str:
                raise ValueError(f"Row {row_index} has empty '{count_column}' field")
            if not embedding_str:
                raise ValueError(
                    f"Row {row_index} has empty '{embedding_column}' field"
                )

            try:
                count = int(count_str)
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_index} has invalid count value: {count_str}"
                ) from exc

            embedding = parse_embedding(embedding_str)

            rows.append(
                {
                    "word": word,
                    "count": count,
                    "embedding": embedding,
                }
            )

    logger.info("Loaded %d rows from CSV", len(rows))
    return rows


def load_model(
    model_path: Path,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    num_quantizers: int,
    codebook_size: int,
    device: torch.device,
) -> RQVAEAutoencoder:
    """Load a trained RQ-VAE model from a checkpoint file.

    Args:
        model_path: Path to the model checkpoint file.
        input_dim: Size of the input feature dimension.
        hidden_dims: Sizes of the hidden layers used in encoder/decoder.
        output_dim: Size of the latent representation before quantization.
        num_quantizers: Number of residual quantizers in the RVQ stack.
        codebook_size: Number of entries per codebook in the RVQ.
        device: Torch device to load the model on.

    Returns:
        Loaded RQ-VAE model in evaluation mode.
    """
    logger.info("Loading model from: %s", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create model instance
    model = RQVAEAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
    )

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded model with %d parameters", num_params)

    return model


def extract_indices(
    model: RQVAEAutoencoder,
    embeddings: list[np.ndarray],
    device: torch.device,
    batch_size: int = 128,
) -> list[list[int]]:
    """Extract quantized indices from embeddings using the trained model.

    Args:
        model: Trained RQ-VAE model.
        embeddings: List of embedding vectors.
        device: Torch device to run inference on.
        batch_size: Batch size for inference.

    Returns:
        List of index lists, where each inner list contains the quantized indices
        for one embedding.
    """
    logger.info("Extracting indices for %d embeddings", len(embeddings))

    all_indices = []

    # Process in batches
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Extracting indices"):
        batch_embeddings = embeddings[i : i + batch_size]

        # Convert to tensor
        batch_tensor = torch.tensor(
            np.stack(batch_embeddings), dtype=torch.float32, device=device
        )

        # Extract indices
        with torch.no_grad():
            indices = model.get_indices(batch_tensor)

        # Convert to list of lists
        indices_cpu = indices.cpu().numpy()
        for row_indices in indices_cpu:
            all_indices.append(row_indices.tolist())

    return all_indices


def save_results_csv(
    output_path: Path,
    words: list[str],
    counts: list[int],
    indices: list[list[int]],
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> None:
    """Save the results to a CSV file, sorted by the elements of the indices.

    Args:
        output_path: Path to the output CSV file.
        words: List of words.
        counts: List of counts.
        indices: List of index lists.
        delimiter: CSV delimiter character.
        encoding: File encoding.
    """
    logger.info("Saving results to CSV: %s", output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine data and sort by indices (lexicographically)
    combined_data = list(zip(words, counts, indices, strict=False))
    combined_data.sort(key=lambda x: x[2])  # Sort by indices (third element)

    logger.info("Sorted %d rows by indices", len(combined_data))

    with output_path.open("w", encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        # Write header
        writer.writerow(["word", "count", "indices"])

        # Write sorted data
        for word, count, index_list in combined_data:
            # Convert indices to JSON string
            indices_str = json.dumps(index_list)
            writer.writerow([word, count, indices_str])

    logger.info("Saved %d rows to CSV", len(combined_data))


def main() -> None:
    """Main function to extract indices from embeddings."""
    parser = argparse.ArgumentParser(
        description="Extract quantized indices from embeddings using a trained RQ-VAE model"
    )

    # Input/output arguments
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to input CSV file containing word, count, and embedding columns",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to output CSV file for word, count, and indices columns",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained RQ-VAE model checkpoint file",
    )

    # CSV format arguments
    parser.add_argument(
        "--word-column",
        type=str,
        default="word",
        help="Name of the word column in input CSV (default: word)",
    )
    parser.add_argument(
        "--count-column",
        type=str,
        default="count",
        help="Name of the count column in input CSV (default: count)",
    )
    parser.add_argument(
        "--embedding-column",
        type=str,
        default="embedding",
        help="Name of the embedding column in input CSV (default: embedding)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter character (default: ,)",
    )

    # Model configuration arguments
    parser.add_argument(
        "--input-dim",
        type=int,
        default=300,
        help="Input dimension of the model (default: 300)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[100, 50],
        help="Hidden layer dimensions (default: 256 64)",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=50,
        help="Output dimension before quantization (default: 64)",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=4,
        help="Number of residual quantizers (default: 8)",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=1000,
        help="Size of each codebook (default: 1024)",
    )

    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (default: cpu)",
    )

    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load embeddings from CSV
    rows = load_embeddings_csv(
        args.input_csv,
        word_column=args.word_column,
        count_column=args.count_column,
        embedding_column=args.embedding_column,
        delimiter=args.delimiter,
    )

    # Extract data
    words = [row["word"] for row in rows]
    counts = [row["count"] for row in rows]
    embeddings = [row["embedding"] for row in rows]

    # Load model
    model = load_model(
        args.model_path,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=args.output_dim,
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        device=device,
    )

    # Extract indices
    indices = extract_indices(model, embeddings, device, args.batch_size)

    # Save results
    save_results_csv(args.output_csv, words, counts, indices, delimiter=args.delimiter)

    logger.info("Index extraction completed successfully")


if __name__ == "__main__":
    main()
