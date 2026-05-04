import networkx as nx

from mdgp.local_search.state import PartitionState, build_partition_state


def split_disconnected_clusters(state: PartitionState) -> bool:
    changed = False

    new_clusters = []

    for cluster in state.clusters:
        if len(cluster) <= 1:
            new_clusters.append(cluster)
            continue

        subgraph = state.G.subgraph(cluster)
        components = list(nx.connected_components(subgraph))

        if len(components) == 1:
            new_clusters.append(cluster)
        else:
            changed = True
            for comp in components:
                new_clusters.append(set(comp))

    if not changed:
        return False

    new_state = build_partition_state(state.G, new_clusters)

    state.clusters = new_state.clusters
    state.cluster_of = new_state.cluster_of
    state.cluster_sizes = new_state.cluster_sizes
    state.internal_edges = new_state.internal_edges

    return True

def min_cut_split_candidate(state: PartitionState, cluster_idx: int):
    cluster = state.clusters[cluster_idx]

    if len(cluster) < 4:
        return None, None, float("-inf")

    H = state.G.subgraph(cluster)

    try:
        cut_value, (a, b) = nx.stoer_wagner(H)
    except Exception:
        return None, None, float("-inf")

    a = set(a)
    b = set(b)

    edges_a = H.subgraph(a).number_of_edges()
    edges_b = H.subgraph(b).number_of_edges()
    edges_ab = state.internal_edges[cluster_idx]

    size_a = len(a)
    size_b = len(b)
    size_ab = state.cluster_sizes[cluster_idx]

    delta = (edges_a / size_a) + (edges_b / size_b) - (edges_ab / size_ab)

    return a, b, delta

def apply_split(state: PartitionState, cluster_idx: int, a: set[int], b: set[int]):
    state.clusters.pop(cluster_idx)
    state.cluster_sizes.pop(cluster_idx)
    state.internal_edges.pop(cluster_idx)

    state.clusters.append(a)
    state.clusters.append(b)

    new_state = build_partition_state(state.G, state.clusters)

    state.clusters = new_state.clusters
    state.cluster_of = new_state.cluster_of
    state.cluster_sizes = new_state.cluster_sizes
    state.internal_edges = new_state.internal_edges


def best_min_cut_split(state: PartitionState):
    best = None
    best_delta = 0.0

    for i in range(len(state.clusters)):
        a, b, delta = min_cut_split_candidate(state, i)

        if a is None:
            continue

        if delta > best_delta:
            best = (i, a, b)
            best_delta = delta

    return best, best_delta