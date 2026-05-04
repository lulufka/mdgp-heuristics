from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import (
    leiden_mdgp_partition,
    leiden_modularity_partition,
)
from mdgp.adapters.leiden_kapoce import leiden_mdgp_kapoce_partition
from mdgp.adapters.matching import matching_partition
from mdgp.config import KAPOCE_CONFIG, KAPOCE_EXECUTABLE
from mdgp.core.evaluation import (
    partition_cluster_sizes,
    partition_density,
    partition_num_clusters,
)
from mdgp.core.graph_io import load_instances
from mdgp.core.types import Partition
from mdgp.local_search.search import (
    LocalSearchResult,
    refine_partition_move_best_improvement,
    refine_partition_move_first_improvement, refine_partition_merge_first_improvement,
    refine_partition_merge_best_improvement, refine_partition_merge_max_intercluster_edges,
    refine_partition_merge_max_boundary_density, refine_partition_split_min_cut,
    refine_partition_star_absorb_singletons, refine_partition_star_form_new_cluster,
)

kapoce_heuristic = partial(
    kapoce_partition,
    executable_path=KAPOCE_EXECUTABLE,
    config_path=KAPOCE_CONFIG,
)


def build_algorithms() -> list[tuple[str, Callable[[nx.Graph], Partition]]]:
    return [
        ("matching", matching_partition),
        ("greedy", greedy_partition),
        ("leiden modularity", leiden_modularity_partition),
        ("leiden mdgp", leiden_mdgp_partition),
        ("leiden with kapoce", leiden_mdgp_kapoce_partition),
        ("kapoce", kapoce_heuristic),
    ]


def safe_percent_improvement(start_score: float, final_score: float) -> float:
    return 100.0 * (final_score - start_score) / start_score


def create_results_dir(base_dir: Path, data_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = Path(data_dir).name
    result_dir = base_dir / "local_search" / f"{dataset_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def run_single_algorithm(
        G: nx.Graph,
        instance_name: str,
        algorithm_name: str,
        algorithm: Callable[[nx.Graph], Partition],
        *,
        postprocessing: str,
        max_passes: int,
        max_moves: int | None,
        random_seed: int | None,
        shuffle_nodes: bool,
) -> dict[str, Any]:
    start_partition = algorithm(G)

    start_score = partition_density(G, start_partition)
    start_num_clusters = partition_num_clusters(start_partition)
    start_sizes = partition_cluster_sizes(start_partition)

    t0 = time.perf_counter()

    if postprocessing == "move_first":
        ls_result: LocalSearchResult = refine_partition_move_first_improvement(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
            random_seed=random_seed,
            shuffle_nodes=shuffle_nodes,
        )
    elif postprocessing == "move_best":
        ls_result = refine_partition_move_best_improvement(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
            random_seed=random_seed,
            shuffle_nodes=shuffle_nodes,
        )
    elif postprocessing == "merge_first":
        ls_result = refine_partition_merge_first_improvement(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    elif postprocessing == "merge_best":
        ls_result = refine_partition_merge_best_improvement(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    elif postprocessing == "merge_max_intercluster_edges":
        ls_result = refine_partition_merge_max_intercluster_edges(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    elif postprocessing == "merge_max_boundary_density":
        ls_result = refine_partition_merge_max_boundary_density(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    elif postprocessing == "split_min_cut":
        ls_result = refine_partition_split_min_cut(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
        )
    elif postprocessing == "star_absorb_singletons":
        ls_result = refine_partition_star_absorb_singletons(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    elif postprocessing == "star_form_new_cluster":
        ls_result = refine_partition_star_form_new_cluster(
            G=G,
            partition=start_partition,
            max_passes=max_passes,
            max_moves=max_moves,
        )
    else:
        raise ValueError(f"Unknown postprocessing: {postprocessing}")

    t1 = time.perf_counter()

    final_partition = ls_result.partition
    final_score = ls_result.final_score
    final_num_clusters = partition_num_clusters(final_partition)
    final_sizes = partition_cluster_sizes(final_partition)

    runtime_ls_ms = (t1 - t0) * 1000.0
    rel_improvement_percent = safe_percent_improvement(start_score, final_score)

    return {
        "instance": instance_name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "algorithm": algorithm_name,
        "postprocessing": postprocessing,
        "start_score": start_score,
        "final_score": final_score,
        "rel_improvement_percent": rel_improvement_percent,
        "start_num_clusters": start_num_clusters,
        "final_num_clusters": final_num_clusters,
        "delta_num_clusters": final_num_clusters - start_num_clusters,
        "start_max_cluster": max(start_sizes) if start_sizes else 0,
        "start_avg_cluster": statistics.mean(start_sizes) if start_sizes else 0.0,
        "final_max_cluster": max(final_sizes) if final_sizes else 0,
        "final_avg_cluster": statistics.mean(final_sizes) if final_sizes else 0.0,
        "runtime_ls_ms": runtime_ls_ms,
        "ls_moves": ls_result.num_moves,
        "ls_passes": ls_result.num_passes,
        "improved": rel_improvement_percent > 0,
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["algorithm", "postprocessing"])
        .agg(
            runs=("algorithm", "size"),
            improved_runs=("improved", "sum"),
            mean_start_score=("start_score", "mean"),
            mean_final_score=("final_score", "mean"),
            mean_rel_improvement_percent=("rel_improvement_percent", "mean"),
            mean_start_num_clusters=("start_num_clusters", "mean"),
            mean_final_num_clusters=("final_num_clusters", "mean"),
            mean_ls_moves=("ls_moves", "mean"),
            mean_ls_passes=("ls_passes", "mean"),
            mean_runtime_ls_ms=("runtime_ls_ms", "mean"),
        )
        .reset_index()
    )

    summary["improved_ratio_percent"] = 100.0 * summary["improved_runs"] / summary["runs"]
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--max-moves", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--shuffle-nodes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_results_dir = Path(args.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    results_dir = create_results_dir(base_results_dir, args.data_dir)

    instances = load_instances(args.data_dir)

    algorithms = build_algorithms()
    postprocessing_methods = [
        "move_first",
        "move_best",
        "merge_first",
        "merge_best",
        "merge_max_intercluster_edges",
        "merge_max_boundary_density",
        "split_min_cut",
        "star_absorb_singletons",
        "star_form_new_cluster",
    ]

    results: list[dict[str, Any]] = []
    total_jobs = len(instances) * len(algorithms) * len(postprocessing_methods)
    current_job = 0

    overall_start = time.perf_counter()

    for instance in instances:
        instance_name = instance.name
        G = instance.G

        print(
            f"\n=== {instance_name} | n={G.number_of_nodes()} | m={G.number_of_edges()} ==="
        )

        for algorithm_name, algorithm in algorithms:
            for postprocessing in postprocessing_methods:
                current_job += 1
                print(
                    f"[{current_job}/{total_jobs}] {algorithm_name} | post={postprocessing}"
                )

                algo_start = time.perf_counter()
                try:
                    result = run_single_algorithm(
                        G=G,
                        instance_name=instance_name,
                        algorithm_name=algorithm_name,
                        algorithm=algorithm,
                        postprocessing=postprocessing,
                        max_passes=args.max_passes,
                        max_moves=args.max_moves,
                        random_seed=args.random_seed,
                        shuffle_nodes=args.shuffle_nodes,
                    )
                    results.append(result)

                except Exception as e:
                    algo_end = time.perf_counter()
                    print(
                        f"    error | {algorithm_name} | post={postprocessing}: {e} ({algo_end - algo_start:.2f}s)"
                    )

    overall_end = time.perf_counter()
    total_runtime_s = overall_end - overall_start

    df = pd.DataFrame(results).round(2)
    summary = build_summary_table(df).round(2)

    detailed_csv_path = results_dir / "detailed_results.csv"
    summary_csv_path = results_dir / "summary_results.csv"
    run_info_path = results_dir / "run_info.txt"

    df.to_csv(detailed_csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    run_info = (
        f"data_dir: {args.data_dir}\n"
        f"results_dir: {results_dir}\n"
        f"postprocessing_methods: {postprocessing_methods}\n"
        f"max_passes: {args.max_passes}\n"
        f"max_moves: {args.max_moves}\n"
        f"random_seed: {args.random_seed}\n"
        f"shuffle_nodes: {args.shuffle_nodes}\n"
        f"num_instances: {len(instances)}\n"
        f"num_algorithms: {len(algorithms)}\n"
        f"total_jobs: {total_jobs}\n"
        f"total_runtime_seconds: {total_runtime_s:.2f}\n"
    )

    write_text_file(run_info_path, run_info)


if __name__ == "__main__":
    main()
