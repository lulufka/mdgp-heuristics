from typing import Optional

from mdgp.local_search.state import PartitionState, build_partition_state, neighbors_in_cluster


def singleton_neighbor_leaves(state: PartitionState, center: int) -> list[int]:
    leaves = []

    center_cluster = state.cluster_of[center]

    for u in state.G.neighbors(center):
        u_cluster = state.cluster_of[u]

        if u_cluster == center_cluster:
            continue

        if state.cluster_sizes[u_cluster] == 1:
            leaves.append(u)

    return leaves

def edges_between_set_and_cluster(state: PartitionState, nodes: set[int], cluster_idx: int) -> int:
    count = 0
    cluster = state.clusters[cluster_idx]

    for u in nodes:
        for v in state.G.neighbors(u):
            if v in cluster:
                count += 1

    return count

def internal_edges_of_set(state: PartitionState, nodes: set[int]) -> int:
    return state.G.subgraph(nodes).number_of_edges()

def delta_absorb_singleton_leaves_into_center_cluster(state: PartitionState, center: int, leaves: set[int]) -> float:
    center_cluster = state.cluster_of[center]

    old_size = state.cluster_sizes[center_cluster]
    old_edges = state.internal_edges[center_cluster]
    old_score = old_edges / old_size

    added_edges_to_cluster = edges_between_set_and_cluster(state, leaves, center_cluster)
    added_edges_inside_leaves = internal_edges_of_set(state, leaves)

    new_size = old_size + len(leaves)
    new_edges = old_edges + added_edges_to_cluster + added_edges_inside_leaves
    new_score = new_edges / new_size

    return new_score - old_score

def apply_absorb_singleton_leaves_into_center_cluster(state: PartitionState, center: int, leaves: set[int]) -> None:
    center_cluster = state.cluster_of[center]

    old_edges = state.internal_edges[center_cluster]
    added_edges_to_cluster = edges_between_set_and_cluster(state, leaves, center_cluster)
    added_edges_inside_leaves = internal_edges_of_set(state, leaves)

    for leaf in leaves:
        leaf_cluster = state.cluster_of[leaf]
        state.clusters[leaf_cluster].remove(leaf)
        state.cluster_sizes[leaf_cluster] = 0
        state.internal_edges[leaf_cluster] = 0

        state.clusters[center_cluster].add(leaf)
        state.cluster_of[leaf] = center_cluster

    state.cluster_sizes[center_cluster] += len(leaves)
    state.internal_edges[center_cluster] = old_edges + added_edges_to_cluster + added_edges_inside_leaves

    delete_empty_singleton_cluster(state)

def delta_form_star_from_center_and_singleton_leaves(state: PartitionState, center: int, leaves: set[int]) -> float:
    source_cluster = state.cluster_of[center]
    source_size = state.cluster_sizes[source_cluster]
    source_edges = state.internal_edges[source_cluster]

    deg_center_sources = neighbors_in_cluster(state, center, source_cluster)

    old_source_score = source_edges / source_size

    if source_size > 1:
        new_source_edges = source_edges - deg_center_sources
        new_source_size = source_size - 1
        new_source_score = new_source_edges / new_source_size
    else:
        new_source_score = 0.0

    star_nodes = set(leaves)
    star_nodes.add(center)

    star_edges = internal_edges_of_set(state, star_nodes)
    star_score = star_edges / len(star_nodes)

    return new_source_score + star_score - old_source_score

def apply_form_star_from_center_and_singleton_leaves(state: PartitionState, center: int, leaves: set[int]) -> None:
    source_cluster = state.cluster_of[center]

    deg_center_source = neighbors_in_cluster(state, center, source_cluster)
    state.clusters[source_cluster].remove(center)
    state.cluster_sizes[source_cluster] -= 1
    state.internal_edges[source_cluster] -= deg_center_source

    for leaf in leaves:
        leaf_cluster = state.cluster_of[leaf]
        state.clusters[leaf_cluster].remove(leaf)
        state.cluster_sizes[leaf_cluster] = 0
        state.internal_edges[leaf_cluster] = 0

    star_cluster = set(leaves)
    star_cluster.add(center)

    state.clusters.append(star_cluster)
    state.cluster_sizes.append(len(star_cluster))
    state.internal_edges.append(internal_edges_of_set(state, star_cluster))

    new_cluster_idx = len(state.clusters) -1
    for node in star_cluster:
        state.cluster_of[node] = new_cluster_idx

    delete_empty_singleton_cluster(state)


def delete_empty_singleton_cluster(state: PartitionState) -> None:
    nonempty_clusters = [set(cluster) for cluster in state.clusters if cluster]
    new_state = build_partition_state(state.G, nonempty_clusters)

    state.clusters = new_state.clusters
    state.cluster_of = new_state.cluster_of
    state.cluster_sizes = new_state.cluster_sizes
    state.internal_edges = new_state.internal_edges


def best_absorb_singleton_leaves_pair(state: PartitionState, min_leaves: int = 2) -> tuple[Optional[tuple[int, set[int]]], float]:
    best: Optional[tuple[int, set[int]]] = None
    best_delta = 0.0

    for center in state.G.nodes():
        leaves = set(singleton_neighbor_leaves(state, center))

        if len(leaves) < min_leaves:
            continue

        delta = delta_absorb_singleton_leaves_into_center_cluster(state, center, leaves)

        if delta > best_delta:
            best_delta = delta
            best = (center, leaves)

    return best, best_delta

def best_form_star_from_singleton_leaves_pair(state: PartitionState, min_leaves: int = 2) -> tuple[Optional[tuple[int, set[int]]], float]:
    best: Optional[tuple[int, set[int]]] = None
    best_delta = 0.0

    for center in state.G.nodes():
        leaves = set(singleton_neighbor_leaves(state, center))

        if len(leaves) < min_leaves:
            continue

        delta = delta_form_star_from_center_and_singleton_leaves(state, center, leaves)

        if delta > best_delta:
            best_delta = delta
            best = (center, leaves)

    return best, best_delta