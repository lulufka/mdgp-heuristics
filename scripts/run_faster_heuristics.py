import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import pandas as pd

from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import leiden_mdgp_partition
from mdgp.adapters.leiden_kapoce import leiden_mdgp_kapoce_partition
from mdgp.adapters.local_search import (
    build_local_search_algorithm,
    build_local_search_portfolio_algorithm,
)
from mdgp.analysis.tables import highlight_density_and_kapoce
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


def build_local_search_experiments() -> list[LocalSearchExperiment]:
    sequential_pipeline = (
        "move_plateau,sparse_vertex_swap,sparse_node_ruin_recreate,"
        "move_plateau,sparse_vertex_swap"
    )

    return [
        LocalSearchExperiment(
            "1 | matching | sequential",
            "matching",
            sequential_pipeline,
        ),
        LocalSearchExperiment(
            "2 | clique seed | sequential",
            "clique_seed",
            sequential_pipeline,
        ),
        LocalSearchExperiment(
            "3 | clique seed | bridge cut vnd",
            "clique_seed",
            "move_plateau,sparse_bridge_singleton_cut,sparse_vertex_swap,"
            "merge_best,sparse_vnd",
        ),
        LocalSearchExperiment(
            "4 | clique seed | exact pair repair",
            "clique_seed",
            "move_plateau,merge_best,sparse_exact_pair_repack,"
            "sparse_bridge_singleton_cut,sparse_vnd",
        ),
        LocalSearchExperiment(
            "5 | clique seed | node ruin repair",
            "clique_seed",
            "move_plateau,sparse_node_ruin_recreate,move_plateau,"
            "sparse_bridge_singleton_cut,merge_best,sparse_vnd",
        ),
        LocalSearchExperiment(
            "6 | clique seed | kapoce-style vns",
            "clique_seed",
            "sparse_kapoce_vns",
        ),
    ]


def build_algorithm_names() -> list[str]:
    return (
            [experiment.name for experiment in build_local_search_experiments()]
            + [
                "kapoce",
            ]
    )


def evaluate_algorithm(
        G: nx.Graph,
        algorithm_name: str,
        algorithm: Callable[[nx.Graph], Partition],
) -> dict[str, Any]:
    start_time = time.time()
    partition = algorithm(G)
    elapsed_time = time.time() - start_time

    cluster_sizes = partition_cluster_sizes(partition)

    return {
        "algorithm": algorithm_name,
        "time": elapsed_time,
        "density": partition_density(G, partition),
        "num": partition_num_clusters(partition),
        "max": max(cluster_sizes),
        "avg": sum(cluster_sizes) / len(cluster_sizes),
    }


def build_algorithm_by_name(
        algorithm_name: str,
        random_seed: int | None,
) -> Callable[[nx.Graph], Partition]:
    experiments = {
        experiment.name: experiment for experiment in build_local_search_experiments()
    }

    if algorithm_name in experiments:
        experiment = experiments[algorithm_name]
        return build_local_search_algorithm(
            experiment.pipeline,
            experiment.start_partition,
            random_seed=random_seed,
        )

    if algorithm_name == "leiden mdgp":
        return partial(leiden_mdgp_partition, random_seed=random_seed)

    if algorithm_name == "kapoce":
        return partial(
            kapoce_partition,
            executable_path=KAPOCE_EXECUTABLE,
            config_path=KAPOCE_CONFIG,
        )

    if algorithm_name == "leiden with kapoce":
        return partial(leiden_mdgp_kapoce_partition, random_seed=random_seed)

    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def evaluate_task(
        instance_name: str,
        G: nx.Graph,
        algorithm_name: str,
        seed: int,
        run_idx: int,
) -> dict[str, Any]:
    algorithm = build_algorithm_by_name(algorithm_name, random_seed=seed)
    result = evaluate_algorithm(G, algorithm_name, algorithm)

    result.update(
        {
            "instance": instance_name,
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "run": run_idx + 1,
            "seed": seed,
        }
    )

    return result


def build_tasks(
        instances: list[Any],
        runs: int,
        base_seed: int,
) -> list[tuple[str, nx.Graph, str, int, int]]:
    tasks = []
    algorithm_names = build_algorithm_names()

    for run_idx in range(runs):
        seed = base_seed + run_idx

        for inst in instances:
            for algorithm_name in algorithm_names:
                tasks.append((inst.name, inst.G, algorithm_name, seed, run_idx))

    return tasks


def run_tasks(
        tasks: list[tuple[str, nx.Graph, str, int, int]],
        workers: int,
        total_runs: int,
) -> list[dict[str, Any]]:
    all_results: list[dict[str, Any]] = []

    if workers <= 1:
        current_run: int | None = None
        current_instance: str | None = None

        for instance_name, G, algorithm_name, seed, run_idx in tasks:
            if current_run != run_idx:
                current_run = run_idx
                print(f"\n=== Random run {run_idx + 1}/{total_runs} | seed={seed} ===")

            if current_instance != instance_name:
                current_instance = instance_name
                print(
                    f"\n[{instance_name}] Start processing "
                    f"(n={G.number_of_nodes()}, m={G.number_of_edges()})"
                )

            result = evaluate_task(instance_name, G, algorithm_name, seed, run_idx)
            print(f"  -> {algorithm_name} finished in {result['time']:.4f}s")
            all_results.append(result)

        return all_results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(evaluate_task, *task): task
            for task in tasks
        }

        for idx, future in enumerate(as_completed(future_to_task), start=1):
            instance_name, _, algorithm_name, seed, run_idx = future_to_task[future]

            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[{idx}/{len(tasks)}] FAILED | "
                    f"run={run_idx + 1}, seed={seed}, "
                    f"instance={instance_name}, algorithm={algorithm_name}: {exc}"
                )
                raise

            print(
                f"[{idx}/{len(tasks)}] "
                f"run={run_idx + 1}, seed={seed} | "
                f"{instance_name} | {algorithm_name} "
                f"finished in {result['time']:.4f}s"
            )
            all_results.append(result)

    return all_results


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
    summary["mean_gap_to_kapoce"] = (
            summary["mean_density"] - summary["mean_kapoce_density"]
    )

    return summary.sort_values(
        ["beats_kapoce", "mean_gap_to_kapoce"],
        ascending=[False, False],
    )


def write_outputs(
        all_results: list[dict[str, Any]],
        results_dir: Path,
        dataset_name: str,
) -> None:
    all_df = pd.DataFrame(all_results)

    all_runs_path = results_dir / f"{dataset_name}_all_random_runs.csv"
    all_df.round(4).to_csv(all_runs_path, index=False)
    print(f"Saved all random runs to {all_runs_path}")

    df = (
        all_df.sort_values(
            ["instance", "algorithm", "density"],
            ascending=[True, True, False],
        )
        .groupby(["instance", "algorithm"], as_index=False)
        .first()
    )

    metrics = ["density", "num", "max", "avg"]

    pivot = df.set_index(["instance", "algorithm"])[metrics].unstack("algorithm")
    pivot = pivot.swaplevel(axis=1).sort_index(axis=1, level=0)

    ordered_columns = []
    for algorithm_name in build_algorithm_names():
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

    try:
        styled = (
            pivot.style.apply(highlight_density_and_kapoce, axis=None)
            .format(precision=1)
            .set_table_styles(
                [
                    {"selector": "table", "props": [("border-collapse", "collapse")]},
                    {
                        "selector": "th",
                        "props": [
                            ("border", "1px solid #999"),
                            ("padding", "4px 6px"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [
                            ("border", "1px solid #ccc"),
                            ("padding", "4px 6px"),
                        ],
                    },
                ]
            )
        )
        styled.to_html(html_path)
        print(f"Saved HTML results to {html_path}")
    except (ImportError, AttributeError) as exc:
        if "jinja2" not in str(exc):
            raise

        pivot.to_html(html_path)
        print(
            f"Saved unstyled HTML results to {html_path} "
            f"because jinja2 is not installed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristics on graph instances.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing graph instances",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset for the output files (defaults to folder name)",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for sequential execution.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name if args.dataset_name else data_dir.name

    instances = load_instances(data_dir)
    tasks = build_tasks(
        instances=instances,
        runs=args.runs,
        base_seed=args.base_seed,
    )

    print(f"Loaded {len(instances)} instances.")
    print(f"Prepared {len(tasks)} evaluation tasks.")
    print(f"Using workers={args.workers}.")

    all_results = run_tasks(
        tasks=tasks,
        workers=args.workers,
        total_runs=args.runs,
    )

    write_outputs(
        all_results=all_results,
        results_dir=results_dir,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()
