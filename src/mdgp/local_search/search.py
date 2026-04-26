import random
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from mdgp.core.evaluation import partition_density
from mdgp.core.types import Partition
from mdgp.local_search.megre import first_improving_merge_pair, apply_merge_clusters, best_merge_pair
from mdgp.local_search.move import best_move_for_node, apply_move_node
from mdgp.local_search.state import build_partition_state


@dataclass
class LocalSearchResult:
    """
    Data class representing the result and statistics of a local search run.
    """
    partition: Partition
    num_moves: int
    num_passes: int
    final_score: float


def refine_partition_move_first_improvement(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 20,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    shuffle_nodes: bool = True,
) -> LocalSearchResult:
    """
    Refines a partition using first-improvement local search.

    Iterates through the nodes (optionally in random order) and immediately applies
    the first node move that improves the partition density.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): The initial partition to refine.
        max_passes (int, optional): Maximum number of passes.
        max_moves (Optional[int], optional): Maximum total node moves allowed.
        random_seed (Optional[int], optional): Seed for randomizing node order.
        shuffle_nodes (bool, optional): Whether to randomize node order per pass.

    Returns:
        LocalSearchResult: An object containing the final partition, total moves applied,
                           total passes used, and the final partition density.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1
        improved_in_pass = False

        nodes = list(G.nodes())
        if shuffle_nodes:
            rng.shuffle(nodes)

        for v in nodes:
            if max_moves is not None and move_count >= max_moves:
                final_partition = [set(cluster) for cluster in state.clusters if cluster]
                return LocalSearchResult(
                    partition=final_partition,
                    num_moves=move_count,
                    num_passes=used_passes,
                    final_score=partition_density(G, final_partition),
                )

            target_cluster, delta = best_move_for_node(state, v)

            if target_cluster is not None and delta > 0:
                apply_move_node(state, v, target_cluster)
                move_count += 1
                improved_in_pass = True

        if not improved_in_pass:
            break

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=move_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_move_best_improvement(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 20,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    shuffle_nodes: bool = True,
) -> LocalSearchResult:
    """
    Refines a partition using best-improvement local search.

    In each pass, all nodes are evaluated and only the single best improving move found in that pass is applied.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): The initial partition to refine.
        max_passes (int, optional): Maximum number of passes.
        max_moves (Optional[int], optional): Maximum total node moves allowed.
        random_seed (Optional[int], optional): Seed for randomizing node order.
        shuffle_nodes (bool, optional): Whether to randomize node order per pass.

    Returns:
        LocalSearchResult: An object containing the final partition, total moves applied,
                           total passes used, and the final partition density.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and move_count >= max_moves:
            break

        used_passes += 1

        nodes = list(G.nodes())
        if shuffle_nodes:
            rng.shuffle(nodes)

        best_v = None
        best_target = None
        best_delta = 0.0

        for v in nodes:
            target_cluster, delta = best_move_for_node(state, v)
            if target_cluster is not None and delta > best_delta:
                best_v = v
                best_target = target_cluster
                best_delta = delta

        if best_v is None or best_target is None or best_delta <= 0:
            break

        apply_move_node(state, best_v, best_target)
        move_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=move_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )

def refine_partition_merge_first_improvement(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 20,
        max_moves: Optional[int] = None
) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and merge_count >= max_moves:
            break

        used_passes += 1

        pair, delta = first_improving_merge_pair(state)

        if pair is None or delta <= 0:
            break

        a, b = pair
        apply_merge_clusters(state, a, b)
        merge_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=merge_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )

def refine_partition_merge_best_improvement(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 20,
        max_moves: Optional[int] = None
) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and merge_count >= max_moves:
            break

        used_passes += 1

        pair, delta = best_merge_pair(state)

        if pair is None or delta <= 0:
            break

        a, b = pair
        apply_merge_clusters(state, a, b)
        merge_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=merge_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )