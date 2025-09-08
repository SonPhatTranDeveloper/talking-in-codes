#!/usr/bin/env python3
"""
Create Embedding Dataset

Reads a CSV file of top words with columns: word,count. Uses the
Word2VecEmbeddingGenerator to compute embeddings for each word and writes a new
parquet file that includes an additional column: embedding.

The embedding is stored as a native array in the parquet format.

Example usage:

    uv run src/scripts/create_embedding_dataset.py \
        --input-csv data/words/words.csv \
        --output-parquet data/embeddings/words.parquet \
        --custom-model models/word2vec/wiki_skipgram_v1.model \
        --on-missing skip
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.utils.embeddings.embedding_generator import (
    Word2VecEmbeddingGenerator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load a top-words CSV file (word,count), generate embeddings, and write a new parquet file"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to input CSV file containing columns: word,count",
    )
    io_group.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Destination parquet path to write with added 'embedding' column",
    )

    model_group = parser.add_argument_group("Embedding Model")
    model_group.add_argument(
        "--custom-model",
        type=Path,
        required=True,
        help="Path to a custom Word2Vec model (.model) trained by our trainer",
    )

    output_group = parser.add_argument_group("Output Formatting")
    output_group.add_argument(
        "--on-missing",
        type=str,
        choices=["skip", "error"],
        default="skip",
        help=(
            "What to do when a word has no embedding in either vocabulary: "
            "'skip' to drop that row, 'error' to raise"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: add embeddings to top-words CSV file and write to a new parquet file.

    Reads the input CSV file containing `word,count`, looks up embeddings for each
    word using the configured generator, and writes a parquet file with columns
    `word,count,embedding`.
    """
    args = parse_arguments()

    logger.info("Starting embedding dataset creation")
    logger.info("=" * 80)
    logger.info(f"  Input CSV:      {args.input_csv}")
    logger.info(f"  Output parquet: {args.output_parquet}")
    logger.info(f"  Custom:         {args.custom_model}")
    logger.info(f"  On-missing:     {args.on_missing}")
    logger.info("=" * 80)

    # Initialize embedding generator
    generator = Word2VecEmbeddingGenerator(
        model_path=str(args.custom_model),
    )

    # Prepare I/O
    input_path: Path = args.input_csv
    output_path: Path = args.output_parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read input CSV file
    logger.info(f"Reading input CSV file: {input_path}")
    df = pd.read_csv(input_path)

    total_rows = len(df)
    logger.info(f"Loaded {total_rows:,} rows from input file")

    # Validate required columns
    if "word" not in df.columns or "count" not in df.columns:
        raise ValueError("Input parquet file must contain 'word' and 'count' columns")

    # Clean data
    df = df.dropna(subset=["word"])
    df["word"] = df["word"].astype(str).str.strip()
    df = df[df["word"] != ""]

    # Convert count to integer, dropping invalid rows
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"])
    df["count"] = df["count"].astype(int)

    cleaned_rows = len(df)
    logger.info(f"After cleaning: {cleaned_rows:,} rows remain")

    # Generate embeddings
    embeddings = []
    skipped_words = []

    for idx, word in enumerate(df["word"]):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"Processed {idx:,}/{cleaned_rows:,} words")

        try:
            vec = generator.get_embedding(word)
            embeddings.append(vec.tolist())  # Convert numpy array to list for parquet
        except KeyError:
            if args.on_missing == "error":
                raise
            logger.debug("Word '%s' not found; skipping", word)
            embeddings.append(None)
            skipped_words.append(idx)

    # Add embeddings to dataframe
    df["embedding"] = embeddings

    # Remove rows with missing embeddings if skip mode
    if args.on_missing == "skip":
        df = df.dropna(subset=["embedding"])

    written_rows = len(df)
    skipped_rows = total_rows - written_rows

    # Write output parquet file
    logger.info(f"Writing output parquet file: {output_path}")
    df.to_parquet(output_path, index=False)

    logger.info("Embedding dataset creation complete")
    logger.info(f"  Total rows read     : {total_rows:,}")
    logger.info(f"  Rows written        : {written_rows:,}")
    logger.info(f"  Rows skipped        : {skipped_rows:,}")
    logger.info(f"  Output parquet saved: {output_path}")


if __name__ == "__main__":
    main()
