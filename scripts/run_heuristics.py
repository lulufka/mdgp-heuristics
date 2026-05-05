import argparse
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import leiden_mdgp_partition
from mdgp.adapters.leiden_kapoce import leiden_mdgp_kapoce_partition
from mdgp.adapters.local_search import build_local_search_algorithm
from mdgp.analysis.tables import highlight_top2_density_multiindex, highlight_beats_kapoce
from mdgp.config import KAPOCE_CONFIG, KAPOCE_EXECUTABLE
from mdgp.core.evaluation import (
    partition_cluster_sizes,
    partition_density,
    partition_num_clusters,
)
from mdgp.core.graph_io import load_instances
from mdgp.core.types import Partition


@dataclass(frozen=True)
class LocalSearchExperiment:
    name: str
    start_partition: str
    pipeline: str


def evaluate_algorithm(
    G: nx.Graph, algorithm_name: str, algorithm: Callable[[nx.Graph], Partition]
) -> dict[str, Any]:
    """
    Evaluates a specific partitioning algorithm on a given graph.

    Args:
        G (nx.Graph): The networkx graph to partition.
        algorithm_name (str): The display name of the algorithm.
        algorithm (Callable): The algorithm function to execute.

    Returns:
        dict[str, Any]: A dictionary containing metrics like execution time, density,
                        cluster count, and cluster size statistics.
    """
    start_time = time.time()
    partition = algorithm(G)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"  -> {algorithm_name} finished in {elapsed_time:.4f}s")

    cluster_sizes = partition_cluster_sizes(partition)

    return {
        "algorithm": algorithm_name,
        "time": elapsed_time,
        "density": partition_density(G, partition),
        "num": partition_num_clusters(partition),
        "max": max(cluster_sizes),
        "avg": sum(cluster_sizes) / len(cluster_sizes),
    }


def build_algorithms() -> list[tuple[str, Callable[[nx.Graph], Partition]]]:
    kapoce_heuristic = partial(
        kapoce_partition,
        executable_path=KAPOCE_EXECUTABLE,
        config_path=KAPOCE_CONFIG,
    )

    local_search_experiments = [
        LocalSearchExperiment("singleton | merge first", "singleton", "merge_first"),
        LocalSearchExperiment("singleton | merge best", "singleton", "merge_best"),
        LocalSearchExperiment(
            "singleton | merge best -> move first",
            "singleton",
            "merge_best,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge first -> move first",
            "singleton",
            "merge_first,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge best -> split min cut -> move first",
            "singleton",
            "merge_best,split_min_cut,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge best -> split min cut -> merge best -> move first",
            "singleton",
            "merge_best,split_min_cut,merge_best,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge max boundary density -> move first",
            "singleton",
            "merge_max_boundary_density,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge max intercluster edges -> move first",
            "singleton",
            "merge_max_intercluster_edges,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge best -> star absorb singletons -> move first",
            "singleton",
            "merge_best,star_absorb_singletons,move_first",
        ),
        LocalSearchExperiment(
            "singleton | merge best -> star form new cluster -> move first",
            "singleton",
            "merge_best,star_form_new_cluster,move_first",
        ),
        LocalSearchExperiment(
            "all in one | split min cut",
            "all_in_one",
            "split_min_cut",
        ),
        LocalSearchExperiment(
            "all in one | split min cut -> move first",
            "all_in_one",
            "split_min_cut,move_first",
        ),
        LocalSearchExperiment(
            "all in one | split min cut -> merge best",
            "all_in_one",
            "split_min_cut,merge_best",
        ),
        LocalSearchExperiment(
            "all in one | split min cut -> merge best -> move first",
            "all_in_one",
            "split_min_cut,merge_best,move_first",
        ),
        LocalSearchExperiment(
            "all in one | split min cut -> star absorb singletons -> move first",
            "all_in_one",
            "split_min_cut,star_absorb_singletons,move_first",
        ),
        LocalSearchExperiment(
            "all in one | split min cut -> star form new cluster -> move first",
            "all_in_one",
            "split_min_cut,star_form_new_cluster,move_first",
        ),
    ]

    local_search_algorithms = [
        (
            experiment.name,
            build_local_search_algorithm(
                experiment.pipeline,
                experiment.start_partition,
            ),
        )
        for experiment in local_search_experiments
    ]

    return [
        *local_search_algorithms,
        ("leiden mdgp", leiden_mdgp_partition),
        ("kapoce", kapoce_heuristic),
        ("leiden with kapoce", leiden_mdgp_kapoce_partition),
    ]


def main() -> None:
    """
    Main entry point for running heuristics on a dataset of graph instances.

    Parses command-line arguments, runs a suite of partitioning algorithms on all
    graphs in the provided directory, and exports the metrics to CSV and HTML tables.
    """
    parser = argparse.ArgumentParser(description="Run heuristics on graph instances.")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing graph instances"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset for the output files (defaults to folder name)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name if args.dataset_name else data_dir.name

    algorithms = build_algorithms()

    instances = load_instances(data_dir)
    results: list[dict[str, Any]] = []

    for inst in instances:
        G = inst.G
        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"\n[{inst.name}] Start processing (n={n}, m={m})")

        for algorithm_name, algorithm in algorithms:
            result = evaluate_algorithm(G, algorithm_name, algorithm)

            result.update({
                "instance": inst.name,
                "n": n,
                "m": m,
            })
            
            results.append(result)

    df = pd.DataFrame(results)
    metrics = ["density", "num", "max", "avg"]

    pivot = df.set_index(["instance", "algorithm"])[metrics].unstack("algorithm")
    pivot = pivot.swaplevel(axis=1).sort_index(axis=1, level=0)

    algorithm_order = [name for name, _ in algorithms]
    ordered_columns = []
    
    for algorithm_name in algorithm_order:
        for metric in metrics:
            col = (algorithm_name, metric)
            if col in pivot.columns:
                ordered_columns.append(col)

    pivot = pivot[ordered_columns].round(1)
    pivot.columns.names = ["algorithm", "metric"]

    csv_path = results_dir / f"{dataset_name}_metrics_table.csv"
    pivot.to_csv(csv_path)
    print(f"\nSaved CSV results to {csv_path}")

    kapoce_summary = build_kapoce_comparison_summary(df).round(2)

    kapoce_summary_path = results_dir / f"{dataset_name}_beats_kapoce_summary.csv"
    kapoce_summary.to_csv(kapoce_summary_path, index=False)
    print(f"Saved KapoCE comparison summary to {kapoce_summary_path}")

    kapoce_summary_html_path = results_dir / f"{dataset_name}_beats_kapoce_summary.html"
    kapoce_summary.to_html(kapoce_summary_html_path, index=False)
    print(f"Saved KapoCE comparison HTML to {kapoce_summary_html_path}")

    html_path = results_dir / f"{dataset_name}_metrics_table.html"
    styled = (
        pivot.style
        .apply(highlight_top2_density_multiindex, axis=None)
        .apply(highlight_beats_kapoce, axis=None)
        .format(precision=1)
        .set_table_styles(
            [
                {"selector": "table", "props": [("border-collapse", "collapse")]},
                {"selector": "th", "props": [("border", "1px solid #999"), ("padding", "4px 6px")]},
                {"selector": "td", "props": [("border", "1px solid #ccc"), ("padding", "4px 6px")]},
            ]
        )
    )
    styled.to_html(html_path)
    print(f"Saved HTML results to {html_path}")

def build_kapoce_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    kapoce_scores = (
        df[df["algorithm"] == "kapoce"]
        .set_index("instance")["density"]
        .rename("kapoce_density")
    )

    comparison = df.join(kapoce_scores, on="instance")
    comparison["beats_kapoce"] = comparison["density"] > comparison["kapoce_density"]
    comparison["ties_kapoce"] = comparison["density"] == comparison["kapoce_density"]

    summary = (
        comparison.groupby("algorithm")
        .agg(
            runs=("instance", "count"),
            beats_kapoce=("beats_kapoce", "sum"),
            ties_kapoce=("ties_kapoce", "sum"),
            mean_density=("density", "mean"),
            mean_kapoce_density=("kapoce_density", "mean"),
        )
        .reset_index()
    )

    summary["beats_kapoce_percent"] = 100 * summary["beats_kapoce"] / summary["runs"]
    summary["mean_gap_to_kapoce"] = summary["mean_density"] - summary["mean_kapoce_density"]

    return summary.sort_values(
        ["beats_kapoce", "mean_gap_to_kapoce"],
        ascending=[False, False],
    )


if __name__ == "__main__":
    main()
