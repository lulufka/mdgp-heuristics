from typing import Optional

from mdgp.local_search.state import PartitionState


def intercluster_edges(state: PartitionState, a: int, b: int) -> int:
    if a == b:
        return 0

    cluster_a = state.clusters[a]
    cluster_b = state.clusters[b]

    if len(cluster_a) > len(cluster_b):
        cluster_a, cluster_b = cluster_b, cluster_a

    cluster_b_set = set(cluster_b)

    count = 0
    for u in cluster_a:
        for v in state.G.neighbors(u):
            if v in cluster_b_set:
                count += 1

    return count

def delta_merge_clusters(state: PartitionState, a: int, b: int) -> float:
    if a == b:
        return float("-inf")

    size_a = state.cluster_sizes[a]
    size_b = state.cluster_sizes[b]

    edges_a = state.internal_edges[a]
    edges_b = state.internal_edges[b]
    edges_ab = intercluster_edges(state, a, b)

    old_score = (edges_a / size_a) + (edges_b / size_b)
    new_score = (edges_a + edges_b + edges_ab) / (size_a + size_b)

    return new_score - old_score

def apply_merge_clusters(state: PartitionState, a: int, b: int) -> None:
    if a == b:
        raise ValueError("Cannot merge the same cluster")

    if a > b:
        a, b = b, a

    edges_a = state.internal_edges[a]
    edges_b = state.internal_edges[b]
    edges_ab = intercluster_edges(state, a, b)

    size_a = state.cluster_sizes[a]
    size_b = state.cluster_sizes[b]

    for node in state.clusters[b]:
        state.cluster_of[node] = a

    state.clusters[a].update(state.clusters[b])

    state.cluster_sizes[a] = size_a + size_b
    state.internal_edges[a] = edges_a + edges_b + edges_ab

    state.clusters.pop(b)
    state.cluster_sizes.pop(b)
    state.internal_edges.pop(b)

    for node, cluster_idx in list(state.cluster_of.items()):
        if cluster_idx > b:
            state.cluster_of[node] = cluster_idx - 1

def neighboring_cluster_pairs(state: PartitionState) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()

    for u, v in state.G.edges():
        cu = state.cluster_of[u]
        cv = state.cluster_of[v]

        if cu != cv:
            a, b = sorted((cu, cv))
            pairs.add((a, b))

    return sorted(pairs)

def best_merge_pair(state: PartitionState) -> tuple[Optional[tuple[int, int]], float]:
    best_pair: Optional[tuple[int, int]] = None
    best_delta = 0.0

    for a, b in neighboring_cluster_pairs(state):
        delta = delta_merge_clusters(state, a, b)
        if delta > best_delta:
            best_delta = delta
            best_pair = (a, b)

    return best_pair, best_delta

def first_improving_merge_pair(state: PartitionState) -> tuple[Optional[tuple[int, int]], float]:
    for a, b in neighboring_cluster_pairs(state):
        delta = delta_merge_clusters(state, a, b)
        if delta > 0:
            return (a, b), delta

    return None, 0.0


def max_intercluster_edges_pair(state: PartitionState) -> tuple[Optional[tuple[int, int]], float]:
    best_pair: Optional[tuple[int, int]] = None
    best_edges = 0
    best_delta = 0.0

    for a, b in neighboring_cluster_pairs(state):
        delta = delta_merge_clusters(state, a, b)
        if delta <= 0:
            continue

        edges_ab = intercluster_edges(state, a, b)
        if edges_ab > best_edges:
            best_edges = edges_ab
            best_pair = (a, b)
            best_delta = delta

    return best_pair, best_delta

def max_boundary_density_pair(state: PartitionState) -> tuple[Optional[tuple[int, int]], float]:
    best_pair: Optional[tuple[int, int]] = None
    best_boundry_density = 0.0
    best_delta = 0.0

    for a, b in neighboring_cluster_pairs(state):
        delta = delta_merge_clusters(state, a, b)
        if delta <= 0:
            continue

        edges_ab = intercluster_edges(state, a, b)
        size_a = state.cluster_sizes[a]
        size_b = state.cluster_sizes[b]

        boundary_density = edges_ab / (size_a * size_b)

        if boundary_density > best_boundry_density:
            best_boundry_density = boundary_density
            best_pair = (a, b)
            best_delta = delta

    return best_pair, best_delta