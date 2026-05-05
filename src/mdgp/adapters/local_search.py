from collections.abc import Callable

import networkx as nx

from mdgp.adapters.initial_partition import get_initial_partitioner
from mdgp.core.types import Partition
from mdgp.local_search.search import (
    LocalSearchResult,
    refine_partition_merge_best_improvement,
    refine_partition_merge_first_improvement,
    refine_partition_merge_max_boundary_density,
    refine_partition_merge_max_intercluster_edges,
    refine_partition_move_first_improvement,
    refine_partition_move_best_improvement,
    refine_partition_split_min_cut,
    refine_partition_star_absorb_singletons,
    refine_partition_star_form_new_cluster,
)

LocalSearchRefiner = Callable[[nx.Graph, Partition], LocalSearchResult]

LOCAL_SEARCH_REFINERS: dict[str, LocalSearchRefiner] = {
    "move_first": refine_partition_move_first_improvement,
    "move_best": refine_partition_move_best_improvement,
    "merge_first": refine_partition_merge_first_improvement,
    "merge_best": refine_partition_merge_best_improvement,
    "merge_max_intercluster_edges": refine_partition_merge_max_intercluster_edges,
    "merge_max_boundary_density": refine_partition_merge_max_boundary_density,
    "split_min_cut": refine_partition_split_min_cut,
    "star_absorb_singletons": refine_partition_star_absorb_singletons,
    "star_form_new_cluster": refine_partition_star_form_new_cluster,
}


def parse_refiners(pipeline: str) -> list[LocalSearchRefiner]:
    step_names = [step.strip() for step in pipeline.split(",") if step.strip()]

    refiners = []
    for step_name in step_names:
        if step_name not in LOCAL_SEARCH_REFINERS:
            known = ", ".join(sorted(LOCAL_SEARCH_REFINERS))
            raise ValueError(
                f"Unknown local-search step '{step_name}'. Known steps: {known}"
            )

        refiners.append(LOCAL_SEARCH_REFINERS[step_name])

    return refiners


def run_local_search_pipeline(
    G: nx.Graph,
    partition: Partition,
    refiners: list[LocalSearchRefiner],
) -> Partition:
    current_partition = partition

    for refine in refiners:
        result = refine(G, current_partition)
        current_partition = result.partition

    return current_partition


def build_local_search_algorithm(
    pipeline: str,
    start_partition: str = "matching",
) -> Callable[[nx.Graph], Partition]:
    initial_partitioner = get_initial_partitioner(start_partition)
    refiners = parse_refiners(pipeline)

    def algorithm(G: nx.Graph) -> Partition:
        initial_partition = initial_partitioner(G)
        return run_local_search_pipeline(G, initial_partition, refiners)

    return algorithm


def build_matching_local_search_algorithm(pipeline: str) -> Callable[[nx.Graph], Partition]:
    return build_local_search_algorithm(pipeline, start_partition="matching")
