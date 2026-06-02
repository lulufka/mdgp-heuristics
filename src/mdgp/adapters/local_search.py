import inspect
from collections.abc import Callable

import networkx as nx

from mdgp.adapters.initial_partition import get_initial_partitioner
from mdgp.core.evaluation import partition_density
from mdgp.core.types import Partition
from mdgp.local_search.search import (
    LocalSearchResult,
    refine_partition_merge_best_improvement,
    refine_partition_merge_first_improvement,
    refine_partition_merge_max_boundary_density,
    refine_partition_merge_max_intercluster_edges,
    refine_partition_move_plateau,
    refine_partition_move_first_improvement,
    refine_partition_move_best_improvement,
    refine_partition_sparse_bridge_split,
    refine_partition_sparse_bridge_singleton_cut,
    refine_partition_sparse_best_peel,
    refine_partition_sparse_exact_pair_repack,
    refine_partition_sparse_exact_multi_repack,
    refine_partition_sparse_exact_small_split,
    refine_partition_sparse_kapoce_vns,
    refine_partition_sparse_low_degree_move,
    refine_partition_sparse_low_degree_peel,
    refine_partition_sparse_node_ruin_recreate,
    refine_partition_sparse_pair_move,
    refine_partition_sparse_ruin_recreate,
    refine_partition_sparse_small_cluster_dissolve,
    refine_partition_sparse_small_cluster_move,
    refine_partition_sparse_vertex_swap,
    refine_partition_sparse_vnd,
    refine_partition_split_min_cut,
    refine_partition_star_absorb_singletons,
    refine_partition_star_form_new_cluster,
)

LocalSearchRefiner = Callable[[nx.Graph, Partition], LocalSearchResult]

LOCAL_SEARCH_REFINERS: dict[str, LocalSearchRefiner] = {
    "move_first": refine_partition_move_first_improvement,
    "move_best": refine_partition_move_best_improvement,
    "move_plateau": refine_partition_move_plateau,
    "merge_best": refine_partition_merge_best_improvement,
    "sparse_bridge_split": refine_partition_sparse_bridge_split,
    "sparse_bridge_singleton_cut": refine_partition_sparse_bridge_singleton_cut,
    "sparse_best_peel": refine_partition_sparse_best_peel,
    "sparse_exact_pair_repack": refine_partition_sparse_exact_pair_repack,
    "sparse_exact_multi_repack": refine_partition_sparse_exact_multi_repack,
    "sparse_exact_small_split": refine_partition_sparse_exact_small_split,
    "sparse_kapoce_vns": refine_partition_sparse_kapoce_vns,
    "sparse_low_degree_move": refine_partition_sparse_low_degree_move,
    "sparse_node_ruin_recreate": refine_partition_sparse_node_ruin_recreate,
    "sparse_small_cluster_dissolve": refine_partition_sparse_small_cluster_dissolve,
    "sparse_pair_move": refine_partition_sparse_pair_move,
    "sparse_ruin_recreate": refine_partition_sparse_ruin_recreate,
    "sparse_vertex_swap": refine_partition_sparse_vertex_swap,
    "sparse_vnd": refine_partition_sparse_vnd,
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
    random_seed: int | None = None,
) -> Partition:
    current_partition = partition

    for refine in refiners:
        kwargs = {}
        parameters = inspect.signature(refine).parameters
        if "random_seed" in parameters:
            kwargs["random_seed"] = random_seed
        if "shuffle_nodes" in parameters:
            kwargs["shuffle_nodes"] = True

        result = refine(G, current_partition, **kwargs)
        current_partition = result.partition

    return current_partition


def build_local_search_algorithm(
    pipeline: str,
    start_partition: str = "matching",
    random_seed: int | None = None,
) -> Callable[[nx.Graph], Partition]:
    initial_partitioner = get_initial_partitioner(start_partition)
    refiners = parse_refiners(pipeline)

    def algorithm(G: nx.Graph) -> Partition:
        kwargs = {}
        parameters = inspect.signature(initial_partitioner).parameters
        if "random_seed" in parameters:
            kwargs["random_seed"] = random_seed

        initial_partition = initial_partitioner(G, **kwargs)
        return run_local_search_pipeline(
            G,
            initial_partition,
            refiners,
            random_seed=random_seed,
        )

    return algorithm


def build_matching_local_search_algorithm(pipeline: str) -> Callable[[nx.Graph], Partition]:
    return build_local_search_algorithm(pipeline, start_partition="matching")


DEFAULT_PORTFOLIO_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "matching",
        "move_first,merge_best,sparse_low_degree_move,sparse_pair_move,"
        "sparse_small_cluster_dissolve,sparse_ruin_recreate,merge_best,"
        "move_first,sparse_bridge_split,merge_best,sparse_exact_small_split,"
        "sparse_best_peel,sparse_vnd",
    ),
    (
        "random_matching",
        "move_first,merge_best,sparse_low_degree_move,sparse_pair_move,"
        "sparse_small_cluster_dissolve,sparse_ruin_recreate,merge_best,"
        "move_first,sparse_bridge_split,merge_best,sparse_exact_small_split,"
        "sparse_best_peel,sparse_vnd",
    ),
    ("matching", "move_first,merge_best,sparse_vnd"),
    ("random_matching", "move_first,merge_best,sparse_vnd"),
    (
        "matching",
        "move_first,merge_best,sparse_low_degree_move,sparse_pair_move,"
        "sparse_small_cluster_dissolve,sparse_ruin_recreate,merge_best,"
        "move_first,sparse_bridge_split,merge_best",
    ),
    (
        "matching",
        "move_first,merge_best,sparse_exact_pair_repack,sparse_exact_small_split,"
        "sparse_best_peel,sparse_vnd",
    ),
    (
        "matching",
        "move_first,merge_best,sparse_exact_pair_repack,sparse_exact_multi_repack,"
        "sparse_exact_small_split,sparse_best_peel,sparse_vnd",
    ),
    ("singleton", "merge_best,sparse_exact_small_split,sparse_best_peel,sparse_vnd"),
)


def build_local_search_portfolio_algorithm(
    candidates: tuple[tuple[str, str], ...] = DEFAULT_PORTFOLIO_CANDIDATES,
    random_seed: int | None = None,
    seed_offsets: tuple[int, ...] = (0,),
) -> Callable[[nx.Graph], Partition]:
    algorithms = []
    for start_partition, pipeline in candidates:
        for offset in seed_offsets:
            candidate_seed = None if random_seed is None else random_seed + offset
            algorithms.append(
                build_local_search_algorithm(
                    pipeline,
                    start_partition=start_partition,
                    random_seed=candidate_seed,
                )
            )

    def algorithm(G: nx.Graph) -> Partition:
        best_partition: Partition | None = None
        best_score = float("-inf")

        for candidate in algorithms:
            partition = candidate(G)
            score = partition_density(G, partition)
            if score > best_score:
                best_partition = partition
                best_score = score

        if best_partition is None:
            return []
        return best_partition

    return algorithm
