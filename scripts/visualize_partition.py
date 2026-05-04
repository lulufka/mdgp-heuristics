from __future__ import annotations

import argparse
import re
from functools import partial
from pathlib import Path
from typing import Callable

import networkx as nx

from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.local_search import build_matching_local_search_algorithm
from mdgp.adapters.matching import matching_partition
from mdgp.analysis.visualization import (
    write_partition_comparison_svg,
    write_partition_svg,
)
from mdgp.core.graph_io import (
    GraphInstance,
    load_instances,
    load_json_graph_instance,
    load_pace_graph_instance,
    load_snap_graph_instance,
)
from mdgp.core.types import Partition


Algorithm = Callable[[nx.Graph], Partition]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize final graph partitions as cluster-colored SVG files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Graph file or directory with .json, .gr, or .txt graph instances.",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name to select when --input is a directory. Defaults to all instances.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        choices=[
            "matching",
            "greedy",
            "leiden_modularity",
            "leiden_mdgp",
            "kapoce",
            "leiden_kapoce",
            "matching_local_search",
        ],
        default=None,
        help="Algorithm to visualize. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="move_first,merge_first,split_min_cut",
        help="Local-search pipeline used with --algorithm matching_local_search.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory for SVG output files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Layout seed. Use the same seed to compare algorithms on identical node positions.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide node labels for larger graphs.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Write one SVG with two selected algorithms side by side. Requires exactly two --algorithm values.",
    )
    return parser.parse_args()


def load_input_instances(path: str | Path) -> list[GraphInstance]:
    path = Path(path)

    if path.is_dir():
        return load_instances(path)

    if path.suffix == ".json":
        return [load_json_graph_instance(path)]
    if path.suffix == ".gr":
        return [load_pace_graph_instance(path)]
    if path.suffix == ".txt":
        return [load_snap_graph_instance(path)]

    raise ValueError(f"Unsupported graph file type: {path}")


def select_instances(
    instances: list[GraphInstance],
    instance_name: str | None,
) -> list[GraphInstance]:
    if instance_name is None:
        return instances

    selected = [instance for instance in instances if instance.name == instance_name]
    if not selected:
        known = ", ".join(instance.name for instance in instances)
        raise ValueError(f"Unknown instance '{instance_name}'. Known instances: {known}")

    return selected


def build_algorithm(name: str, pipeline: str) -> Algorithm:
    if name == "matching":
        return matching_partition
    if name == "greedy":
        return greedy_partition
    if name == "leiden_modularity":
        from mdgp.adapters.external.leiden import leiden_modularity_partition

        return leiden_modularity_partition
    if name == "leiden_mdgp":
        from mdgp.adapters.external.leiden import leiden_mdgp_partition

        return leiden_mdgp_partition
    if name == "leiden_kapoce":
        from mdgp.adapters.leiden_kapoce import leiden_mdgp_kapoce_partition

        return leiden_mdgp_kapoce_partition
    if name == "matching_local_search":
        return build_matching_local_search_algorithm(pipeline)
    if name == "kapoce":
        from mdgp.adapters.external.kapoce import kapoce_partition
        from mdgp.config import KAPOCE_CONFIG, KAPOCE_EXECUTABLE

        return partial(
            kapoce_partition,
            executable_path=KAPOCE_EXECUTABLE,
            config_path=KAPOCE_CONFIG,
        )

    raise ValueError(f"Unknown algorithm: {name}")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def display_name(algorithm_name: str, pipeline: str) -> str:
    if algorithm_name == "matching_local_search":
        return f"{algorithm_name} | {pipeline}"
    return algorithm_name


def write_single_algorithm_svg(
    instance: GraphInstance,
    algorithm_name: str,
    algorithm: Algorithm,
    output_dir: Path,
    pipeline: str,
    seed: int,
    show_labels: bool,
) -> Path:
    partition = algorithm(instance.G)
    filename = safe_filename(f"{instance.name}_{algorithm_name}.svg")
    output_path = output_dir / filename
    title = f"{instance.name} | {display_name(algorithm_name, pipeline)}"

    write_partition_svg(
        instance.G,
        partition,
        output_path,
        title=title,
        seed=seed,
        show_labels=show_labels,
    )
    return output_path


def write_comparison_svg(
    instance: GraphInstance,
    algorithm_names: list[str],
    output_dir: Path,
    pipeline: str,
    seed: int,
    show_labels: bool,
) -> Path:
    if len(algorithm_names) != 2:
        raise ValueError("--compare requires exactly two --algorithm values.")

    left_name, right_name = algorithm_names
    left_partition = build_algorithm(left_name, pipeline)(instance.G)
    right_partition = build_algorithm(right_name, pipeline)(instance.G)
    filename = safe_filename(f"{instance.name}_{left_name}_vs_{right_name}.svg")
    output_path = output_dir / filename

    write_partition_comparison_svg(
        instance.G,
        left_partition,
        right_partition,
        output_path,
        title=f"{instance.name} | partition comparison",
        left_title=display_name(left_name, pipeline),
        right_title=display_name(right_name, pipeline),
        seed=seed,
        show_labels=show_labels,
    )
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    algorithm_names = args.algorithm or ["matching", "matching_local_search", "kapoce"]

    instances = select_instances(load_input_instances(args.input), args.instance)

    for instance in instances:
        print(
            f"\n[{instance.name}] n={instance.G.number_of_nodes()} m={instance.G.number_of_edges()}"
        )

        if args.compare:
            output_path = write_comparison_svg(
                instance,
                algorithm_names,
                output_dir,
                args.pipeline,
                args.seed,
                not args.hide_labels,
            )
            print(f"  wrote {output_path}")
            continue

        for algorithm_name in algorithm_names:
            algorithm = build_algorithm(algorithm_name, args.pipeline)
            output_path = write_single_algorithm_svg(
                instance,
                algorithm_name,
                algorithm,
                output_dir,
                args.pipeline,
                args.seed,
                not args.hide_labels,
            )
            print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
