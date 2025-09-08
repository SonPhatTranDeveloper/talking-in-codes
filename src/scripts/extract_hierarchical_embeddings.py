#!/usr/bin/env python3
"""
Extract Multi-Level Hierarchical Embeddings

Reads a parquet file containing word embeddings, performs hierarchical clustering
on the embeddings, and extracts multiple levels of flat clusters using fcluster.
Saves the results to a CSV file with word and discrete embedding arrays representing
cluster assignments at different hierarchical levels.

The script supports various linkage methods and distance metrics for hierarchical
clustering, and allows specification of multiple cluster levels simultaneously.
The output includes a discrete embedding array where each position represents
the cluster assignment at that level.

Example usage:

    uv run src/scripts/extract_hierarchical_embeddings.py \
        --input-parquet data/embeddings/words.parquet \
        --output-csv data/clusters/clusters.csv \
        --cluster-levels 20 200 1000 170000 \
        --linkage ward \
        --metric euclidean
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

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
            "Load embedding parquet file, perform hierarchical clustering, "
            "and save clusters to CSV"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to input parquet file containing word embeddings",
    )
    io_group.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to output CSV file for cluster results",
    )

    clustering_group = parser.add_argument_group("Clustering Parameters")
    clustering_group.add_argument(
        "--cluster-levels",
        type=int,
        nargs="+",
        help="List of cluster levels to extract (e.g., 2 3 4 10 50 100)",
    )
    clustering_group.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help="List of distance thresholds for cluster extraction (mutually exclusive with --cluster-levels)",
    )
    clustering_group.add_argument(
        "--linkage",
        type=str,
        choices=[
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ],
        default="ward",
        help="Linkage method for hierarchical clustering",
    )
    clustering_group.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for clustering (e.g., euclidean, cosine, manhattan)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include original embeddings in the output CSV",
    )
    output_group.add_argument(
        "--save-dendrogram",
        type=Path,
        help="Optional path to save dendrogram plot as PNG",
    )
    output_group.add_argument(
        "--ensure-unique",
        action="store_true",
        help="Ensure discrete embeddings are unique by adding word frequency as tie-breaker",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If arguments are invalid or conflicting.
    """
    if args.cluster_levels is None and args.thresholds is None:
        raise ValueError("Must specify either --cluster-levels or --thresholds")

    if args.cluster_levels is not None and args.thresholds is not None:
        raise ValueError("Cannot specify both --cluster-levels and --thresholds")

    if args.cluster_levels is not None:
        if any(n < 1 for n in args.cluster_levels):
            raise ValueError("All cluster levels must be positive")
        if len(set(args.cluster_levels)) != len(args.cluster_levels):
            raise ValueError("Cluster levels must be unique")

    if args.thresholds is not None:
        if any(t <= 0 for t in args.thresholds):
            raise ValueError("All thresholds must be positive")

    if args.linkage == "ward" and args.metric != "euclidean":
        raise ValueError("Ward linkage requires euclidean metric")


def load_embeddings(input_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load embeddings from parquet file.

    Args:
        input_path: Path to input parquet file.

    Returns:
        Tuple of (dataframe with metadata, embedding matrix).
    """
    logger.info(f"Loading embeddings from: {input_path}")
    df = pd.read_parquet(input_path)

    # Validate required columns
    if "word" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Input parquet must contain 'word' and 'embedding' columns")

    # Convert embeddings to numpy array
    embeddings = np.array(df["embedding"].tolist())

    logger.info(
        f"Loaded {len(df):,} words with {embeddings.shape[1]}-dimensional embeddings"
    )

    return df, embeddings


def perform_hierarchical_clustering(
    embeddings: np.ndarray, linkage_method: str, metric: str
) -> np.ndarray:
    """Perform hierarchical clustering on embeddings.

    Args:
        embeddings: Matrix of embeddings (n_samples x n_features).
        linkage_method: Linkage method for clustering.
        metric: Distance metric to use.

    Returns:
        Linkage matrix from hierarchical clustering.
    """
    logger.info(f"Computing pairwise distances using {metric} metric")

    if linkage_method == "ward":
        # Ward linkage uses euclidean distance internally
        linkage_matrix = linkage(embeddings, method=linkage_method)
    else:
        # Compute distance matrix first for other methods
        distances = pdist(embeddings, metric=metric)
        linkage_matrix = linkage(distances, method=linkage_method)

    logger.info(f"Performed hierarchical clustering with {linkage_method} linkage")

    return linkage_matrix


def extract_multi_level_clusters(
    linkage_matrix: np.ndarray,
    cluster_levels: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Extract multiple levels of flat clusters from hierarchical clustering.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering.
        cluster_levels: List of cluster numbers to extract.
        thresholds: List of distance thresholds for cluster extraction.

    Returns:
        Dictionary mapping level names to cluster assignment arrays.
    """
    cluster_results = {}

    if cluster_levels is not None:
        logger.info(f"Extracting clusters for levels: {cluster_levels}")
        for n_clusters in sorted(cluster_levels):
            clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
            n_extracted = len(np.unique(clusters))
            level_name = f"cluster_level_{n_clusters}"
            cluster_results[level_name] = clusters
            logger.info(f"Level {n_clusters}: extracted {n_extracted} unique clusters")

    else:  # thresholds is not None
        logger.info(f"Extracting clusters for thresholds: {thresholds}")
        for i, threshold in enumerate(sorted(thresholds, reverse=True)):
            clusters = fcluster(linkage_matrix, threshold, criterion="distance")
            n_extracted = len(np.unique(clusters))
            level_name = f"threshold_level_{i + 1}"
            cluster_results[level_name] = clusters
            logger.info(
                f"Threshold {threshold}: extracted {n_extracted} unique clusters"
            )

    return cluster_results


def save_dendrogram(linkage_matrix: np.ndarray, output_path: Path) -> None:
    """Save dendrogram plot to file.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering.
        output_path: Path to save the dendrogram PNG file.
    """
    try:
        import matplotlib.pyplot as plt

        logger.info("Generating dendrogram plot")
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode="level", p=10)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index or (Cluster Size)")
        plt.ylabel("Distance")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Dendrogram saved to: {output_path}")

    except ImportError:
        logger.warning("matplotlib not available; skipping dendrogram generation")


def save_multi_level_results(
    df: pd.DataFrame,
    cluster_results: dict[str, np.ndarray],
    output_path: Path,
    include_embeddings: bool = False,
) -> None:
    """Save multi-level clustering results to CSV file with discrete embeddings.

    Args:
        df: Original dataframe with word metadata.
        cluster_results: Dictionary mapping level names to cluster assignments.
        output_path: Path to output CSV file.
        include_embeddings: Whether to include original embeddings in output.
    """
    logger.info("Preparing multi-level output data with discrete embeddings")

    # Create output dataframe
    output_df = df[["word"]].copy()
    if "count" in df.columns:
        output_df["count"] = df["count"]

    # Create discrete embedding array from cluster assignments
    # Sort cluster results by the numeric part of level names for consistent ordering
    sorted_levels = sorted(
        cluster_results.items(),
        key=lambda x: int(x[0].split("_")[-1])
        if x[0].startswith("cluster_level_")
        else float(x[0].split("_")[-1]),
    )

    discrete_embeddings = []
    level_names = []

    for level_name, clusters in sorted_levels:
        discrete_embeddings.append(clusters)
        level_names.append(level_name)

    # Convert to array where each row is a word's discrete embedding
    discrete_embedding_matrix = np.array(discrete_embeddings).T

    # Convert each row to a list for CSV storage
    output_df["discrete_embedding"] = [
        row.tolist() for row in discrete_embedding_matrix
    ]

    if include_embeddings:
        output_df["original_embedding"] = df["embedding"]

    # Sort by the discrete embedding array content (hierarchical ordering)
    # Create sort keys for each level of the discrete embedding
    for i in range(discrete_embedding_matrix.shape[1]):
        output_df[f"_sort_key_{i}"] = discrete_embedding_matrix[:, i]

    # Build sort columns: all discrete embedding levels, then count if available
    sort_columns = [f"_sort_key_{i}" for i in range(discrete_embedding_matrix.shape[1])]
    sort_ascending = [True] * len(sort_columns)

    if "count" in output_df.columns:
        sort_columns.append("count")
        sort_ascending.append(False)  # Count in descending order

    output_df = output_df.sort_values(sort_columns, ascending=sort_ascending)

    # Remove temporary sort columns
    sort_key_columns = [
        f"_sort_key_{i}" for i in range(discrete_embedding_matrix.shape[1])
    ]
    output_df = output_df.drop(columns=sort_key_columns)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info(f"Multi-level results saved to: {output_path}")
    logger.info(f"Output contains {len(output_df):,} words with discrete embeddings")
    logger.info(
        f"Discrete embedding dimensions: {len(level_names)} (levels: {[name.split('_')[-1] for name in level_names]})"
    )

    # Log statistics for each level
    for i, (level_name, clusters) in enumerate(sorted_levels):
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        n_clusters = len(cluster_counts)
        logger.info(f"Level {i + 1} ({level_name}): {n_clusters} clusters")
        logger.info(f"  Min size: {cluster_counts.min()}")
        logger.info(f"  Max size: {cluster_counts.max()}")
        logger.info(f"  Mean size: {cluster_counts.mean():.1f}")


def main() -> None:
    """Entry point: perform hierarchical clustering on embeddings and save results.

    Reads embeddings from parquet file, performs hierarchical clustering,
    extracts flat clusters, and saves results to CSV with cluster assignments.
    """
    args = parse_arguments()
    validate_arguments(args)

    logger.info("Starting multi-level hierarchical embedding clustering")
    logger.info("=" * 80)
    logger.info(f"  Input parquet:    {args.input_parquet}")
    logger.info(f"  Output CSV:       {args.output_csv}")
    logger.info(f"  Linkage method:   {args.linkage}")
    logger.info(f"  Distance metric:  {args.metric}")
    if args.cluster_levels:
        logger.info(f"  Cluster levels:   {args.cluster_levels}")
    else:
        logger.info(f"  Distance thresholds: {args.thresholds}")
    logger.info(f"  Include embeddings: {args.include_embeddings}")
    if args.save_dendrogram:
        logger.info(f"  Save dendrogram:  {args.save_dendrogram}")
    logger.info("=" * 80)

    # Load embeddings
    df, embeddings = load_embeddings(args.input_parquet)

    # Perform hierarchical clustering
    linkage_matrix = perform_hierarchical_clustering(
        embeddings, args.linkage, args.metric
    )

    # Extract multi-level flat clusters
    cluster_results = extract_multi_level_clusters(
        linkage_matrix, args.cluster_levels, args.thresholds
    )

    # Save dendrogram if requested
    if args.save_dendrogram:
        save_dendrogram(linkage_matrix, args.save_dendrogram)

    # Save multi-level results
    save_multi_level_results(
        df, cluster_results, args.output_csv, args.include_embeddings
    )

    logger.info("Hierarchical clustering completed successfully")


if __name__ == "__main__":
    main()
