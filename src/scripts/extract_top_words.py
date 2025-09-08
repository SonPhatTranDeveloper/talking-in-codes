#!/usr/bin/env python3
"""
Extract top-K most frequent words from WikiText datasets and save to CSV.

This script uses the WikiLoader with an NLTK word tokenizer and a lowercase
word processor to produce tokenized sentences, counts word frequencies, filters
by a minimum count, selects the top-K frequent words, and writes the results to
CSV in the format: word,count.

Example usage:

    uv run src/scripts/extract_top_words.py \
        --dataset-name wikitext-103-v1 \
        --splits train validation test \
        --min-count 5 \
        --top-k 200000 \
        --output-csv data/words/words.csv
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from src.utils.dataset.wiki_loader import LowercaseWordProcessor, WikiLoader
from src.utils.tokenizer.word_tokenizer import NLTKWordTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for extracting top words.

    Returns:
        argparse.Namespace: Parsed arguments including dataset configuration,
            frequency filtering, selection count, and output path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract top-K frequent words from WikiText datasets and save to CSV"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset configuration
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
        help="Dataset splits to include",
    )

    # Extraction configuration
    extract_group = parser.add_argument_group("Extraction Configuration")
    extract_group.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum frequency required for a word to be considered",
    )
    extract_group.add_argument(
        "--top-k",
        type=int,
        default=10000,
        help=(
            "Number of top frequent words to output. Use -1 to output all that meet min-count"
        ),
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-csv",
        type=Path,
        default=Path("models/top_words.csv"),
        help="Destination CSV file path (word,count)",
    )

    return parser.parse_args()


def write_word_counts_to_csv(rows: list[tuple[str, int]], output_path: Path) -> None:
    """Write word counts to a CSV file with header `word,count`.

    Args:
        rows: List of (word, count) tuples to write.
        output_path: Destination path for the CSV file. Parent directories will
            be created if they do not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count"])  # header
        writer.writerows(rows)


def main() -> None:
    """Entry point: extract and save top-K words for the configured dataset."""
    args = parse_arguments()

    logger.info("Starting top words extraction")
    logger.info("=" * 80)
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Splits: {', '.join(args.splits)}")
    logger.info(f"  Min count: {args.min_count}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Output CSV: {args.output_csv}")
    logger.info("=" * 80)

    # Initialize data loader (NLTK tokenizer + lowercase word processor)
    wiki_loader = WikiLoader(
        dataset_name=args.dataset_name,
        splits=tuple(args.splits),
        word_tokenizer=NLTKWordTokenizer(),
        word_processor=LowercaseWordProcessor(),
    )

    logger.info("Loading and tokenizing dataset...")
    sentences = wiki_loader.load_data()
    if not sentences:
        logger.error("No data loaded. Exiting without writing CSV.")
        return
    logger.info(f"Loaded {len(sentences):,} sentences")

    # Count word frequencies incrementally to limit memory usage
    counter: Counter[str] = Counter()
    for sentence in sentences:
        if sentence:
            counter.update(sentence)

    logger.info(f"Counted {len(counter):,} unique words before filtering")

    # Filter by minimum count
    filtered_items = [(w, c) for w, c in counter.items() if c >= args.min_count]
    logger.info(f"{len(filtered_items):,} words meet min_count >= {args.min_count}")

    # Sort by frequency descending
    filtered_items.sort(key=lambda x: x[1], reverse=True)

    # Select top-K if requested
    if args.top_k is not None and args.top_k >= 0:
        top_items = filtered_items[: args.top_k] if args.top_k > 0 else filtered_items
    else:
        top_items = filtered_items

    logger.info(f"Writing {len(top_items):,} word counts to CSV")
    write_word_counts_to_csv(top_items, args.output_csv)

    logger.info("Extraction completed successfully")
    logger.info(f"CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
