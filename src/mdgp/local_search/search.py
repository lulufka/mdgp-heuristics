import random
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from mdgp.core.evaluation import partition_density
from mdgp.core.types import Partition
from mdgp.local_search.merge import first_improving_merge_pair, apply_merge_clusters, best_merge_pair, \
    max_intercluster_edges_pair, max_boundary_density_pair
from mdgp.local_search.move import (
    apply_isolate_node,
    apply_move_node,
    best_move_for_node,
    delta_isolate_node,
    delta_move_node,
)
from mdgp.local_search.split import split_disconnected_clusters, best_min_cut_split, apply_split
from mdgp.local_search.sparse import (
    apply_exact_repack,
    apply_bridge_singleton_cut,
    apply_peel_node_as_singleton,
    apply_move_node_set,
    apply_rebuilt_clusters,
    apply_swap_nodes,
    apply_small_cluster_move,
    best_exact_small_split,
    best_exact_multi_repack,
    best_exact_pair_repack,
    best_bridge_singleton_cut,
    best_pair_move,
    best_bridge_split,
    best_low_degree_move,
    best_node_ruin_and_recreate,
    best_peel_node,
    best_low_degree_peel,
    best_ruin_and_recreate,
    best_small_cluster_dissolve,
    best_small_cluster_move,
    best_vertex_swap,
    node_ruin_and_recreate_candidate,
)
from mdgp.local_search.star import best_absorb_singleton_leaves_pair, apply_absorb_singleton_leaves_into_center_cluster, \
    best_form_star_from_singleton_leaves_pair, apply_form_star_from_center_and_singleton_leaves
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
    max_passes: int = 2000,
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
    max_passes: int = 2000,
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


def refine_partition_move_plateau(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 100,
    max_moves: Optional[int] = None,
    max_zero_gain_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    shuffle_nodes: bool = True,
    epsilon: float = 1e-12,
) -> LocalSearchResult:
    """
    Label-propagation style node moves with bounded zero-gain plateau walking.

    For each node this considers neighboring clusters and the option to isolate the
    node as a singleton. Improving moves are always accepted; zero-gain moves are
    accepted up to max_zero_gain_moves to avoid unbounded cycling on plateaus.
    """
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    zero_gain_count = 0
    zero_gain_limit = (
        max_zero_gain_moves
        if max_zero_gain_moves is not None
        else max(1, 2 * G.number_of_nodes())
    )
    used_passes = 0

    for _ in range(max_passes):
        used_passes += 1
        changed_in_pass = False

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

            source_cluster = state.cluster_of[v]
            candidate_clusters = {
                state.cluster_of[u]
                for u in state.G.neighbors(v)
                if state.cluster_of[u] != source_cluster
            }

            best_target: int | None = None
            best_is_singleton = False
            best_delta = float("-inf")

            for target_cluster in candidate_clusters:
                delta = delta_move_node(state, v, target_cluster)
                if delta > best_delta:
                    best_target = target_cluster
                    best_is_singleton = False
                    best_delta = delta

            isolate_delta = delta_isolate_node(state, v)
            if isolate_delta > best_delta:
                best_target = None
                best_is_singleton = True
                best_delta = isolate_delta

            is_improving = best_delta > epsilon
            is_zero_gain = abs(best_delta) <= epsilon
            if not is_improving and not (
                is_zero_gain and zero_gain_count < zero_gain_limit
            ):
                continue

            if best_is_singleton:
                apply_isolate_node(state, v)
            elif best_target is not None:
                apply_move_node(state, v, best_target)
            else:
                continue

            move_count += 1
            changed_in_pass = True
            if is_zero_gain:
                zero_gain_count += 1

        if not changed_in_pass:
            break

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
        max_passes: int = 2000,
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
        max_passes: int = 2000,
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

def refine_partition_merge_max_intercluster_edges(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 2000,
        max_moves: Optional[int] = None
) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and merge_count >= max_moves:
            break

        used_passes += 1

        pair, delta = max_intercluster_edges_pair(state)

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

def refine_partition_merge_max_boundary_density(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 2000,
        max_moves: Optional[int] = None
) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    merge_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and merge_count >= max_moves:
            break

        used_passes += 1

        pair, delta = max_boundary_density_pair(state)

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


def refine_partition_split_min_cut(
        G: nx.Graph,
        partition: Partition,
        max_passes: int = 2000,
        max_moves: Optional[int] = None
) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    split_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        used_passes += 1

        best, delta = best_min_cut_split(state)

        if best is None or delta <= 0:
            break

        cluster_idx, a, b = best
        apply_split(state, cluster_idx, a, b)
        split_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=split_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )

def refine_partition_star_absorb_singletons(G: nx.Graph, partition: Partition, max_passes: int = 2000, max_moves: Optional[int] = None, min_leaves: int = 2) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    operation_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1

        candidate, delta = best_absorb_singleton_leaves_pair(state, min_leaves)

        if candidate is None or delta <= 0:
            break

        center, leaves = candidate
        apply_absorb_singleton_leaves_into_center_cluster(state, center, leaves)
        operation_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]

    return LocalSearchResult(
        partition=final_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition)
    )

def refine_partition_star_form_new_cluster(G: nx.Graph, partition: Partition, max_passes: int = 2000, max_moves: Optional[int] = None, min_leaves: int = 2) -> LocalSearchResult:
    state = build_partition_state(G, partition)

    operation_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1

        candidate, delta = best_form_star_from_singleton_leaves_pair(state, min_leaves)

        if candidate is None or delta <= 0:
            break

        center, leaves = candidate
        apply_form_star_from_center_and_singleton_leaves(state, center, leaves)
        operation_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]

    return LocalSearchResult(
        partition=final_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition)
    )


def refine_partition_sparse_bridge_split(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    split_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and split_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        best, delta = best_bridge_split(state, random_seed=step_seed)

        if best is None or delta <= 0:
            break

        cluster_idx, a, b = best
        apply_split(state, cluster_idx, a, b)
        split_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=split_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_bridge_singleton_cut(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    min_remaining_size: int = 3,
    epsilon: float = 1e-12,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    cut_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and cut_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        cut, delta = best_bridge_singleton_cut(
            state,
            min_remaining_size=min_remaining_size,
            random_seed=step_seed,
            epsilon=epsilon,
        )

        if cut is None or delta < -epsilon:
            break

        node, target = cut
        apply_bridge_singleton_cut(state, node, target)
        cut_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=cut_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_low_degree_peel(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_internal_degree: int = 1,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    peel_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and peel_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        node, delta = best_low_degree_peel(
            state,
            max_internal_degree=max_internal_degree,
            random_seed=step_seed,
        )

        if node is None or delta <= 0:
            break

        apply_peel_node_as_singleton(state, node)
        peel_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=peel_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_best_peel(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    peel_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and peel_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        node, delta = best_peel_node(state, random_seed=step_seed)

        if node is None or delta <= 0:
            break

        apply_peel_node_as_singleton(state, node)
        peel_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=peel_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_exact_small_split(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_cluster_size: int = 10,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    split_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and split_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        best, delta = best_exact_small_split(
            state,
            max_cluster_size=max_cluster_size,
            random_seed=step_seed,
        )

        if best is None or delta <= 0:
            break

        cluster_idx, a, b = best
        apply_split(state, cluster_idx, a, b)
        split_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=split_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_exact_pair_repack(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 50,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_nodes: int = 12,
    max_cluster_size: int = 6,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    repack_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and repack_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        best, delta = best_exact_pair_repack(
            state,
            max_nodes=max_nodes,
            max_cluster_size=max_cluster_size,
            random_seed=step_seed,
        )

        if best is None or delta <= 0:
            break

        replaced_indices, new_clusters = best
        apply_exact_repack(state, replaced_indices, new_clusters)
        repack_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=repack_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_exact_multi_repack(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 5,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    group_size: int = 3,
    max_nodes: int = 10,
    max_cluster_size: int = 6,
    max_neighbors_per_cluster: int = 4,
    max_groups: int = 60,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    repack_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and repack_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        best, delta = best_exact_multi_repack(
            state,
            group_size=group_size,
            max_nodes=max_nodes,
            max_cluster_size=max_cluster_size,
            max_neighbors_per_cluster=max_neighbors_per_cluster,
            max_groups=max_groups,
            random_seed=step_seed,
        )

        if best is None or delta <= 0:
            break

        replaced_indices, new_clusters = best
        apply_exact_repack(state, replaced_indices, new_clusters)
        repack_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=repack_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_low_degree_move(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_graph_degree: int = 4,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and move_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        node, target, delta = best_low_degree_move(
            state,
            max_graph_degree=max_graph_degree,
            random_seed=step_seed,
        )

        if node is None or target is None or delta <= 0:
            break

        apply_move_node(state, node, target)
        move_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=move_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_small_cluster_move(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_cluster_size: int = 5,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and move_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        pair, delta = best_small_cluster_move(
            state,
            max_cluster_size=max_cluster_size,
            random_seed=step_seed,
        )

        if pair is None or delta <= 0:
            break

        source, target = pair
        apply_small_cluster_move(state, source, target)
        move_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=move_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_small_cluster_dissolve(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_cluster_size: int = 3,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    dissolve_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and dissolve_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        clusters, delta = best_small_cluster_dissolve(
            state,
            max_cluster_size=max_cluster_size,
            random_seed=step_seed,
        )

        if clusters is None or delta <= 0:
            break

        apply_rebuilt_clusters(state, clusters)
        dissolve_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=dissolve_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_pair_move(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    move_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and move_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        move, delta = best_pair_move(state, random_seed=step_seed)

        if move is None or delta <= 0:
            break

        nodes, target = move
        apply_move_node_set(state, nodes, target)
        move_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=move_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_vertex_swap(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    swap_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and swap_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        swap, delta = best_vertex_swap(
            state,
            random_seed=step_seed,
            max_candidates=max_candidates,
        )

        if swap is None or delta <= 0:
            break

        u, v = swap
        apply_swap_nodes(state, u, v)
        swap_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=swap_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_node_ruin_recreate(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 5,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    fraction: float = 0.05,
    attempts: int = 3,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    operation_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        clusters, delta = best_node_ruin_and_recreate(
            state,
            fraction=fraction,
            attempts=attempts,
            random_seed=step_seed,
        )

        if clusters is None or delta <= 0:
            break

        apply_rebuilt_clusters(state, clusters)
        operation_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def refine_partition_sparse_ruin_recreate(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    fraction: float = 0.10,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    operation_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1
        step_seed = rng.randrange(2**32) if random_seed is not None else None
        clusters, delta = best_ruin_and_recreate(
            state,
            fraction=fraction,
            random_seed=step_seed,
        )

        if clusters is None or delta <= 0:
            break

        apply_rebuilt_clusters(state, clusters)
        operation_count += 1

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )


def _kapoce_like_intensify(
    G: nx.Graph,
    partition: Partition,
    rng: random.Random,
    random_seed: Optional[int],
) -> Partition:
    current = partition

    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_move_plateau(
        G,
        current,
        max_passes=10,
        max_zero_gain_moves=max(1, min(G.number_of_nodes(), 500)),
        random_seed=step_seed,
    ).partition

    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_sparse_vertex_swap(
        G,
        current,
        max_passes=25,
        max_candidates=20000,
        random_seed=step_seed,
    ).partition

    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_sparse_bridge_singleton_cut(
        G,
        current,
        max_passes=50,
        random_seed=step_seed,
    ).partition

    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_sparse_exact_small_split(
        G,
        current,
        max_passes=20,
        random_seed=step_seed,
    ).partition

    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_sparse_small_cluster_dissolve(
        G,
        current,
        max_passes=20,
        random_seed=step_seed,
    ).partition

    current = refine_partition_merge_best_improvement(G, current).partition
    step_seed = rng.randrange(2**32) if random_seed is not None else None
    current = refine_partition_move_best_improvement(
        G,
        current,
        max_passes=100,
        random_seed=step_seed,
    ).partition

    return current


def refine_partition_sparse_kapoce_vns(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 5,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    shake_fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25),
) -> LocalSearchResult:
    """
    KaPoCE-inspired VNS: intensify, shake by node ruin/recreate, intensify again.

    Shaking candidates may be worse before the second intensification. The refiner
    returns the best partition found, so the pipeline remains monotone overall.
    """
    rng = random.Random(random_seed)

    current = _kapoce_like_intensify(G, partition, rng, random_seed)
    best_partition = current
    best_score = partition_density(G, best_partition)

    operation_count = 0
    used_passes = 0

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1
        improved = False

        for fraction in shake_fractions:
            if max_moves is not None and operation_count >= max_moves:
                break

            state = build_partition_state(G, current)
            step_seed = rng.randrange(2**32) if random_seed is not None else None
            shaken, _ = node_ruin_and_recreate_candidate(
                state,
                fraction=fraction,
                random_seed=step_seed,
            )
            if shaken is None:
                continue

            candidate = _kapoce_like_intensify(G, shaken, rng, random_seed)
            candidate_score = partition_density(G, candidate)
            operation_count += 1

            if candidate_score > best_score:
                current = candidate
                best_partition = candidate
                best_score = candidate_score
                improved = True
                break

        if not improved:
            break

    return LocalSearchResult(
        partition=best_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=best_score,
    )


def refine_partition_sparse_vnd(
    G: nx.Graph,
    partition: Partition,
    max_passes: int = 2000,
    max_moves: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_cluster_size: int = 10,
) -> LocalSearchResult:
    rng = random.Random(random_seed)
    state = build_partition_state(G, partition)

    operation_count = 0
    used_passes = 0

    split_disconnected_clusters(state)

    for _ in range(max_passes):
        if max_moves is not None and operation_count >= max_moves:
            break

        used_passes += 1
        improved = False

        step_seed = rng.randrange(2**32) if random_seed is not None else None
        bridge_cut, bridge_cut_delta = best_bridge_singleton_cut(
            state,
            random_seed=step_seed,
        )
        if bridge_cut is not None and bridge_cut_delta >= 0:
            node, target = bridge_cut
            apply_bridge_singleton_cut(state, node, target)
            operation_count += 1
            improved = True

        if improved:
            continue

        step_seed = rng.randrange(2**32) if random_seed is not None else None
        best_split, split_delta = best_exact_small_split(
            state,
            max_cluster_size=max_cluster_size,
            random_seed=step_seed,
        )
        if best_split is not None and split_delta > 0:
            cluster_idx, a, b = best_split
            apply_split(state, cluster_idx, a, b)
            operation_count += 1
            improved = True

        if improved:
            continue

        step_seed = rng.randrange(2**32) if random_seed is not None else None
        node, peel_delta = best_peel_node(state, random_seed=step_seed)
        if node is not None and peel_delta > 0:
            apply_peel_node_as_singleton(state, node)
            operation_count += 1
            improved = True

        if improved:
            continue

        step_seed = rng.randrange(2**32) if random_seed is not None else None
        pair_move, pair_delta = best_pair_move(state, random_seed=step_seed)
        if pair_move is not None and pair_delta > 0:
            nodes, target = pair_move
            apply_move_node_set(state, nodes, target)
            operation_count += 1
            improved = True

        if improved:
            continue

        nodes = list(G.nodes())
        if random_seed is not None:
            rng.shuffle(nodes)

        best_v = None
        best_target = None
        best_delta = 0.0
        for v in nodes:
            target, delta = best_move_for_node(state, v)
            if target is not None and delta > best_delta:
                best_v = v
                best_target = target
                best_delta = delta

        if best_v is not None and best_target is not None and best_delta > 0:
            apply_move_node(state, best_v, best_target)
            operation_count += 1
            improved = True

        if improved:
            continue

        step_seed = rng.randrange(2**32) if random_seed is not None else None
        rebuilt, dissolve_delta = best_small_cluster_dissolve(
            state,
            random_seed=step_seed,
        )
        if rebuilt is not None and dissolve_delta > 0:
            apply_rebuilt_clusters(state, rebuilt)
            operation_count += 1
            improved = True

        if improved:
            continue

        pair, merge_delta = best_merge_pair(state)
        if pair is not None and merge_delta > 0:
            a, b = pair
            apply_merge_clusters(state, a, b)
            operation_count += 1
            improved = True

        if not improved:
            break

    final_partition = [set(cluster) for cluster in state.clusters if cluster]
    return LocalSearchResult(
        partition=final_partition,
        num_moves=operation_count,
        num_passes=used_passes,
        final_score=partition_density(G, final_partition),
    )
