from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.local_search import build_local_search_algorithm
from mdgp.config import KAPOCE_CONFIG, KAPOCE_EXECUTABLE
from mdgp.core.evaluation import (
    partition_cluster_sizes,
    partition_density,
    partition_num_clusters,
)
from mdgp.core.graph_io import GraphInstance, load_instances
from mdgp.core.types import Partition


@dataclass(frozen=True)
class PipelineExperiment:
    name: str
    start_partition: str
    pipeline: str


DEFAULT_SPARSE_DATA_DIRS = [
    "data/er_graphs/small_sparse",
    "data/partition_graphs/small_sparse",
    "data/powerlaw_graphs/small_sparse",
]

BUILD_STEPS = [
    "merge_best",
    "merge_first",
    "merge_max_boundary_density",
    "merge_max_intercluster_edges",
    "merge_max_boundary_density",
    "sparse_small_cluster_move",
    "sparse_low_degree_move",
]

IMPROVE_STEPS = [
    "move_first",
    "move_best",
    "merge_best",
    "star_absorb_singletons",
    "star_form_new_cluster",
    "sparse_pair_move",
    "sparse_ruin_recreate",
    "sparse_low_degree_move",
]

REPAIR_STEPS = [
    "sparse_bridge_split",
    "sparse_low_degree_peel",
    "sparse_small_cluster_dissolve",
    "split_min_cut",
]

patterns = [
    ["build", "improve", "repair"],
    ["build", "improve", "improve", "repair"],
    ["build", "repair", "improve", "repair"],
    ["build", "improve", "repair", "improve", "repair"],
    ["build", "improve", "repair", "improve", "repair", "improve", "repair"],
]

step_groups = {
    "build": BUILD_STEPS,
    "improve": IMPROVE_STEPS,
    "repair": REPAIR_STEPS,
}

DEFAULT_EXPERIMENTS = [
    PipelineExperiment(
        "singleton | merge best -> sparse bridge split -> merge best -> sparse low degree move",
        "singleton",
        "merge_best,sparse_bridge_split,merge_best,sparse_low_degree_move",
    ),
    PipelineExperiment(
        "singleton | merge best -> sparse small cluster dissolve -> sparse pair move -> sparse ruin recreate -> move first",
        "singleton",
        "merge_best,sparse_small_cluster_dissolve,sparse_pair_move,sparse_ruin_recreate,move_first",
    ),
    PipelineExperiment(
        "matching | sparse pair move -> sparse small cluster move -> sparse ruin recreate -> merge best",
        "matching",
        "sparse_pair_move,sparse_small_cluster_move,sparse_ruin_recreate,merge_best",
    ),
    PipelineExperiment(
        "singleton | merge max boundary density -> sparse low degree move -> sparse bridge split -> merge best",
        "singleton",
        "merge_max_boundary_density,sparse_low_degree_move,sparse_bridge_split,merge_best",
    ),
    PipelineExperiment(
        "singleton | merge max intercluster edges -> sparse low degree move -> merge best -> move first",
        "singleton",
        "merge_max_intercluster_edges,sparse_low_degree_move,merge_best,move_first",
    ),
    PipelineExperiment(
        "singleton | merge best -> move first -> sparse low degree peel -> sparse bridge split -> merge best",
        "singleton",
        "merge_best,move_first,sparse_low_degree_peel,sparse_bridge_split,merge_best",
    ),
    PipelineExperiment(
        "matching | sparse low degree move -> merge best -> sparse bridge split -> move first",
        "matching",
        "sparse_low_degree_move,merge_best,sparse_bridge_split,move_first",
    ),
    PipelineExperiment(
        "matching | move first -> sparse bridge split -> merge best -> sparse low degree move",
        "matching",
        "move_first,sparse_bridge_split,merge_best,sparse_low_degree_move",
    ),
    PipelineExperiment(
        "all in one | sparse bridge split -> sparse low degree peel -> merge best -> move first",
        "all_in_one",
        "sparse_bridge_split,sparse_low_degree_peel,merge_best,move_first",
    ),
    PipelineExperiment(
        "singleton | merge best -> move best -> split min cut -> sparse bridge split -> merge best",
        "singleton",
        "merge_best,move_best,split_min_cut,sparse_bridge_split,merge_best",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run randomized local-search pipeline combinations on sparse graph datasets."
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        default=None,
        help="Sparse graph directory. Can be passed multiple times.",
    )
    parser.add_argument("--results-dir", type=str, default="results/sparse_local_search")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--random-pipelines", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--start-partition",
        action="append",
        choices=["singleton", "matching", "all_in_one"],
        default=None,
        help="Start partition for randomized pipelines. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-avg-degree",
        type=float,
        default=None,
        help="Skip instances with average degree above this threshold.",
    )
    parser.add_argument("--skip-kapoce", action="store_true")
    return parser.parse_args()


def existing_default_data_dirs() -> list[str]:
    return [data_dir for data_dir in DEFAULT_SPARSE_DATA_DIRS if Path(data_dir).exists()]


def load_all_sparse_instances(data_dirs: list[str]) -> list[tuple[str, GraphInstance]]:
    instances = []
    for data_dir in data_dirs:
        dataset_name = Path(data_dir).name
        parent_name = Path(data_dir).parent.name
        for instance in load_instances(data_dir):
            instances.append((f"{parent_name}/{dataset_name}", instance))
    return instances


def generate_random_experiments(
    *,
    count: int,
    seed: int,
    min_steps: int,
    max_steps: int,
    start_partitions: list[str],
) -> list[PipelineExperiment]:
    rng = random.Random(seed)
    experiments: list[PipelineExperiment] = []
    seen: set[tuple[str, str]] = set()

    while len(experiments) < count:
        pattern = rng.choice(patterns)

        if len(pattern) < min_steps or len(pattern) > max_steps:
            continue

        steps = [rng.choice(step_groups[group]) for group in pattern]

        start_partition = rng.choice(start_partitions)
        pipeline = ",".join(steps)
        key = (start_partition, pipeline)
        if key in seen:
            continue

        seen.add(key)
        label = pipeline.replace(",", " -> ")
        experiments.append(
            PipelineExperiment(
                f"{start_partition} | random {len(experiments) + 1:02d} | {label}",
                start_partition,
                pipeline,
            )
        )

    return experiments


def evaluate_algorithm(
    G: nx.Graph,
    algorithm: Callable[[nx.Graph], Partition],
) -> dict[str, Any]:
    start = time.perf_counter()
    partition = algorithm(G)
    elapsed = time.perf_counter() - start
    cluster_sizes = partition_cluster_sizes(partition)

    return {
        "density": partition_density(G, partition),
        "num_clusters": partition_num_clusters(partition),
        "max_cluster": max(cluster_sizes) if cluster_sizes else 0,
        "avg_cluster": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
        "runtime_s": elapsed,
    }


def build_kapoce_algorithm() -> Callable[[nx.Graph], Partition]:
    return partial(
        kapoce_partition,
        executable_path=KAPOCE_EXECUTABLE,
        config_path=KAPOCE_CONFIG,
    )


def summarize_against_kapoce(df: pd.DataFrame) -> pd.DataFrame:
    if "kapoce_density" not in df.columns or df["kapoce_density"].isna().all():
        return pd.DataFrame()

    comparable = df[
        (df["algorithm"] != "kapoce") & df["kapoce_density"].notna()
    ].copy()
    if comparable.empty:
        return pd.DataFrame()

    comparable["beats_kapoce"] = comparable["density"] > comparable["kapoce_density"]
    comparable["gap_to_kapoce"] = comparable["density"] - comparable["kapoce_density"]

    summary = (
        comparable.groupby(["algorithm", "start_partition", "pipeline"])
        .agg(
            runs=("instance", "count"),
            beats_kapoce=("beats_kapoce", "sum"),
            mean_density=("density", "mean"),
            mean_kapoce_density=("kapoce_density", "mean"),
            mean_gap_to_kapoce=("gap_to_kapoce", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
        )
        .reset_index()
    )
    summary["beats_kapoce_percent"] = 100.0 * summary["beats_kapoce"] / summary["runs"]

    return summary.sort_values(
        ["beats_kapoce", "mean_gap_to_kapoce", "mean_density"],
        ascending=[False, False, False],
    )


def main() -> None:
    args = parse_args()

    data_dirs = args.data_dir or existing_default_data_dirs()
    if not data_dirs:
        raise ValueError("No sparse data directories found. Pass --data-dir explicitly.")

    start_partitions = args.start_partition or ["singleton", "matching", "all_in_one"]
    random_experiments = generate_random_experiments(
        count=args.random_pipelines,
        seed=args.base_seed,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        start_partitions=start_partitions,
    )
    experiments = [*DEFAULT_EXPERIMENTS, *random_experiments]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    instances = load_all_sparse_instances(data_dirs)
    if args.max_avg_degree is not None:
        instances = [
            (dataset, instance)
            for dataset, instance in instances
            if instance.G.number_of_nodes() == 0
            or (2 * instance.G.number_of_edges() / instance.G.number_of_nodes())
            <= args.max_avg_degree
        ]

    kapoce_algorithm = None if args.skip_kapoce else build_kapoce_algorithm()
    rows: list[dict[str, Any]] = []

    total_jobs = len(instances) * args.runs * len(experiments)
    if kapoce_algorithm is not None:
        total_jobs += len(instances)

    job = 0
    kapoce_scores: dict[str, float] = {}

    for dataset, instance in instances:
        G = instance.G
        instance_key = f"{dataset}/{instance.name}"
        avg_degree = 0.0 if G.number_of_nodes() == 0 else 2 * G.number_of_edges() / G.number_of_nodes()

        print(
            f"\n=== {instance_key} | n={G.number_of_nodes()} | m={G.number_of_edges()} | avg_deg={avg_degree:.2f} ===",
            flush=True,
        )

        if kapoce_algorithm is not None:
            job += 1
            print(f"[{job}/{total_jobs}] kapoce", flush=True)
            kapoce_result = evaluate_algorithm(G, kapoce_algorithm)
            kapoce_scores[instance_key] = kapoce_result["density"]
            rows.append(
                {
                    "dataset": dataset,
                    "instance": instance_key,
                    "n": G.number_of_nodes(),
                    "m": G.number_of_edges(),
                    "avg_degree": avg_degree,
                    "run": 0,
                    "seed": None,
                    "algorithm": "kapoce",
                    "start_partition": "",
                    "pipeline": "",
                    "kapoce_density": kapoce_result["density"],
                    **kapoce_result,
                }
            )

        for run_idx in range(args.runs):
            seed = args.base_seed + run_idx
            for experiment in experiments:
                job += 1
                print(f"[{job}/{total_jobs}] seed={seed} | {experiment.name}", flush=True)

                algorithm = build_local_search_algorithm(
                    experiment.pipeline,
                    start_partition=experiment.start_partition,
                    random_seed=seed,
                )
                result = evaluate_algorithm(G, algorithm)
                kapoce_density = kapoce_scores.get(instance_key)

                rows.append(
                    {
                        "dataset": dataset,
                        "instance": instance_key,
                        "n": G.number_of_nodes(),
                        "m": G.number_of_edges(),
                        "avg_degree": avg_degree,
                        "run": run_idx + 1,
                        "seed": seed,
                        "algorithm": experiment.name,
                        "start_partition": experiment.start_partition,
                        "pipeline": experiment.pipeline,
                        "kapoce_density": kapoce_density,
                        **result,
                    }
                )

    df = pd.DataFrame(rows)
    detailed_path = results_dir / "sparse_local_search_detailed.csv"
    df.round(4).to_csv(detailed_path, index=False)

    summary = summarize_against_kapoce(df)
    if not summary.empty:
        summary_path = results_dir / "sparse_local_search_beats_kapoce.csv"
        summary.round(4).to_csv(summary_path, index=False)

    best_path = results_dir / "sparse_local_search_best_by_instance.csv"
    best = (
        df[df["algorithm"] != "kapoce"]
        .sort_values(["instance", "density"], ascending=[True, False])
        .groupby("instance")
        .head(10)
    )
    best.round(4).to_csv(best_path, index=False)

    pipelines_path = results_dir / "sparse_local_search_pipelines.txt"
    pipeline_lines = [
        f"{experiment.name}\n  start={experiment.start_partition}\n  pipeline={experiment.pipeline}"
        for experiment in experiments
    ]
    pipelines_path.write_text("\n\n".join(pipeline_lines), encoding="utf-8")

    print(f"\nSaved detailed results to {detailed_path.resolve()}")
    if not summary.empty:
        print(f"Saved KapoCE comparison to {summary_path.resolve()}")
    print(f"Saved best-by-instance table to {best_path.resolve()}")
    print(f"Saved pipeline manifest to {pipelines_path.resolve()}")


if __name__ == "__main__":
    main()
