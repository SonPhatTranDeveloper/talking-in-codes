#!/usr/bin/env python3
"""
Extract Multi-Level Bisecting K-Means Embeddings

Reads a parquet file containing word embeddings, performs bisecting k-means clustering
on the embeddings, and extracts multiple levels of flat clusters at different k values.
Saves the results to a CSV file with word and discrete embedding arrays representing
cluster assignments at different cluster levels.

The script supports various distance metrics for bisecting k-means clustering,
and allows specification of multiple cluster levels simultaneously.
The output includes a discrete embedding array where each position represents
the cluster assignment at that level.

Caching Support:
The script supports caching of the computationally expensive clustering models to
speed up subsequent runs with the same embeddings and clustering parameters.
Use --cache-dir to enable caching, and --force-recompute to bypass cache.

Example usage:

    # Basic usage
    uv run src/scripts/extract_hierarchical_embeddings_bisecting_k_means.py \
        --input-parquet data/embeddings/words.parquet \
        --output-csv data/clusters/clusters.csv \
        --cluster-levels 20 200 1000 \
        --metric euclidean

    # With caching enabled and unique embeddings
    uv run src/scripts/extract_hierarchical_embeddings_bisecting_k_means.py \
        --input-parquet data/embeddings/words.parquet \
        --output-csv data/clusters/clusters.csv \
        --cluster-levels 1000 1000 1000 \
        --metric euclidean \
        --cache-dir cache/bisecting_kmeans_models \
        --ensure-unique
"""

import argparse
import hashlib
import logging
import pickle
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
            "Load embedding parquet file, perform bisecting k-means clustering, "
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
        required=True,
        help="List of cluster levels to extract (e.g., 2 3 4 10 50 100)",
    )
    clustering_group.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for clustering (euclidean, cosine)",
    )
    clustering_group.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of random initializations for k-means",
    )
    clustering_group.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum number of iterations for k-means",
    )
    clustering_group.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducible results",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include original embeddings in the output CSV",
    )
    output_group.add_argument(
        "--ensure-unique",
        action="store_true",
        help="Ensure discrete embeddings are unique by adding group-based sequential indices (0,1,2,...) for identical embeddings",
    )

    cache_group = parser.add_argument_group("Caching Options")
    cache_group.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory to store cached clustering models (enables caching if provided)",
    )
    cache_group.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cached result exists",
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If arguments are invalid or conflicting.
    """
    if any(n < 1 for n in args.cluster_levels):
        raise ValueError("All cluster levels must be positive")
    if len(set(args.cluster_levels)) != len(args.cluster_levels):
        raise ValueError("Cluster levels must be unique")

    if args.metric not in ["euclidean", "cosine"]:
        raise ValueError("Metric must be either 'euclidean' or 'cosine'")

    if args.n_init < 1:
        raise ValueError("n_init must be positive")

    if args.max_iter < 1:
        raise ValueError("max_iter must be positive")


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


def generate_cache_key(
    embeddings: np.ndarray, metric: str, n_init: int, max_iter: int, random_state: int
) -> str:
    """Generate a unique cache key for the clustering parameters.

    Args:
        embeddings: Matrix of embeddings (n_samples x n_features).
        metric: Distance metric to use.
        n_init: Number of random initializations.
        max_iter: Maximum number of iterations.
        random_state: Random state for reproducibility.

    Returns:
        Unique cache key string.
    """
    # Create a hash based on embeddings shape, data hash, and parameters
    embeddings_hash = hashlib.md5(embeddings.tobytes()).hexdigest()
    param_string = f"{metric}_{n_init}_{max_iter}_{random_state}_{embeddings.shape[0]}_{embeddings.shape[1]}"
    cache_key = f"bisecting_kmeans_{param_string}_{embeddings_hash[:16]}"
    return cache_key


def save_linkage_matrix(
    linkage_matrix: np.ndarray, cache_dir: Path, cache_key: str
) -> None:
    """Save linkage matrix to cache directory.

    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering.
        cache_dir: Directory to store cached results.
        cache_key: Unique identifier for this clustering configuration.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"

    cache_data = {
        "linkage_matrix": linkage_matrix,
        "cache_key": cache_key,
        "timestamp": pd.Timestamp.now(),
        "shape": linkage_matrix.shape,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    file_size = cache_file.stat().st_size / (1024 * 1024)  # Size in MB
    logger.info(f"Linkage matrix cached to: {cache_file}")
    logger.info(f"Cache file size: {file_size:.2f} MB")


def load_linkage_matrix(cache_dir: Path, cache_key: str) -> np.ndarray | None:
    """Load linkage matrix from cache directory.

    Args:
        cache_dir: Directory containing cached results.
        cache_key: Unique identifier for this clustering configuration.

    Returns:
        Cached linkage matrix if found and valid, None otherwise.
    """
    cache_file = cache_dir / f"{cache_key}.pkl"

    if not cache_file.exists():
        logger.info(f"No cached result found for key: {cache_key}")
        return None

    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        # Validate cache data structure
        if not isinstance(cache_data, dict) or "linkage_matrix" not in cache_data:
            logger.warning(f"Invalid cache file format: {cache_file}")
            return None

        linkage_matrix = cache_data["linkage_matrix"]
        cached_timestamp = cache_data.get("timestamp", "unknown")

        logger.info(f"Loaded cached linkage matrix from: {cache_file}")
        logger.info(f"Cache timestamp: {cached_timestamp}")
        logger.info(f"Matrix shape: {linkage_matrix.shape}")

        return linkage_matrix

    except (pickle.PickleError, EOFError, KeyError) as e:
        logger.warning(f"Failed to load cache file {cache_file}: {e}")
        return None


def perform_hierarchical_clustering(
    embeddings: np.ndarray,
    linkage_method: str,
    metric: str,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Perform hierarchical clustering on embeddings with optional caching.

    Args:
        embeddings: Matrix of embeddings (n_samples x n_features).
        linkage_method: Linkage method for clustering.
        metric: Distance metric to use.
        cache_dir: Optional directory to store/load cached results.
        force_recompute: Force recomputation even if cached result exists.

    Returns:
        Linkage matrix from hierarchical clustering.
    """
    # Try to load from cache if caching is enabled
    if cache_dir is not None and not force_recompute:
        cache_key = generate_cache_key(embeddings, linkage_method, metric)
        cached_matrix = load_linkage_matrix(cache_dir, cache_key)
        if cached_matrix is not None:
            logger.info("Using cached linkage matrix")
            return cached_matrix

    # Compute linkage matrix
    logger.info(f"Computing pairwise distances using {metric} metric")
    logger.info(
        f"Processing {embeddings.shape[0]:,} samples with {embeddings.shape[1]} dimensions"
    )

    if linkage_method == "ward":
        # Ward linkage uses euclidean distance internally
        linkage_matrix = linkage(embeddings, method=linkage_method)
    else:
        # Compute distance matrix first for other methods
        distances = pdist(embeddings, metric=metric)
        linkage_matrix = linkage(distances, method=linkage_method)

    logger.info(f"Performed hierarchical clustering with {linkage_method} linkage")

    # Save to cache if caching is enabled
    if cache_dir is not None:
        cache_key = generate_cache_key(embeddings, linkage_method, metric)
        save_linkage_matrix(linkage_matrix, cache_dir, cache_key)

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
    ensure_unique: bool = False,
) -> None:
    """Save multi-level clustering results to CSV file with discrete embeddings.

    Args:
        df: Original dataframe with word metadata.
        cluster_results: Dictionary mapping level names to cluster assignments.
        output_path: Path to output CSV file.
        include_embeddings: Whether to include original embeddings in output.
        ensure_unique: Whether to ensure discrete embeddings are unique by adding group-based sequential indices.
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

    # Ensure uniqueness if requested
    if ensure_unique:
        logger.info(
            "Ensuring discrete embeddings are unique using group-based sequential indexing"
        )

        # Check for duplicates before processing
        unique_before = len(np.unique(discrete_embedding_matrix, axis=0))
        total_embeddings = len(discrete_embedding_matrix)

        logger.info(f"Total embeddings: {total_embeddings:,}")
        logger.info(f"Unique embeddings before indexing: {unique_before:,}")

        if unique_before < total_embeddings:
            logger.info(
                f"Found {total_embeddings - unique_before:,} duplicate embeddings to resolve"
            )

            # Create a DataFrame to work with embeddings and their indices
            embedding_df = pd.DataFrame(discrete_embedding_matrix)

            # Create a string representation of each embedding for grouping
            embedding_strings = embedding_df.apply(
                lambda row: "_".join(map(str, row)), axis=1
            )

            # Group by embedding and assign sequential indices within each group
            group_indices = embedding_strings.groupby(embedding_strings).cumcount()

            # Add the group index as an additional dimension
            group_indices_array = group_indices.values.reshape(-1, 1)
            discrete_embedding_matrix = np.hstack(
                [discrete_embedding_matrix, group_indices_array]
            )
            level_names.append("group_index")

            # Verify uniqueness after adding group indices
            unique_after = len(np.unique(discrete_embedding_matrix, axis=0))

            logger.info(f"Unique embeddings after group indexing: {unique_after:,}")

            if unique_after == total_embeddings:
                logger.info("✓ All embeddings are now unique")
            else:
                logger.warning(
                    f"⚠ Still have {total_embeddings - unique_after:,} duplicate embeddings after group indexing"
                )

            # Log group statistics
            group_sizes = embedding_strings.value_counts()
            num_groups = len(group_sizes)
            max_group_size = group_sizes.max()
            avg_group_size = group_sizes.mean()

            logger.info("Group statistics:")
            logger.info(f"  Total groups: {num_groups:,}")
            logger.info(f"  Largest group size: {max_group_size:,}")
            logger.info(f"  Average group size: {avg_group_size:.2f}")
            logger.info(f"  Groups with duplicates: {(group_sizes > 1).sum():,}")

        else:
            logger.info("✓ All embeddings are already unique, no indexing needed")

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
    logger.info(f"  Ensure unique:    {args.ensure_unique}")
    if args.save_dendrogram:
        logger.info(f"  Save dendrogram:  {args.save_dendrogram}")
    if args.cache_dir:
        logger.info(f"  Cache directory:  {args.cache_dir}")
        logger.info(f"  Force recompute:  {args.force_recompute}")
    logger.info("=" * 80)

    # Load embeddings
    df, embeddings = load_embeddings(args.input_parquet)

    # Perform hierarchical clustering
    linkage_matrix = perform_hierarchical_clustering(
        embeddings, args.linkage, args.metric, args.cache_dir, args.force_recompute
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
        df,
        cluster_results,
        args.output_csv,
        args.include_embeddings,
        args.ensure_unique,
    )

    logger.info("Hierarchical clustering completed successfully")


if __name__ == "__main__":
    main()
