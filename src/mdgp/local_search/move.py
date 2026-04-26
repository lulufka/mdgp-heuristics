from typing import Optional

from mdgp.local_search.state import PartitionState, neighbors_in_cluster


def delta_move_node(state: PartitionState, v: int, target_cluster: int) -> float:
    """
    Calculates the change in the total partition density if node v is moved.

    Args:
        state (PartitionState): The current state of the partition.
        v (int): The node to be moved.
        target_cluster (int): The index of the destination cluster.

    Returns:
        float: The change in density. A positive value means an improvement.
               Returns negative infinity if the move is invalid (e.g., same cluster
               or moving would result in an empty cluster).
    """
    source_cluster = state.cluster_of[v]

    if source_cluster == target_cluster:
        return float("-inf")

    source_size = state.cluster_sizes[source_cluster]
    target_size = state.cluster_sizes[target_cluster]

    source_edges = state.internal_edges[source_cluster]
    target_edges = state.internal_edges[target_cluster]

    deg_source = neighbors_in_cluster(state, v, source_cluster)
    deg_target = neighbors_in_cluster(state, v, target_cluster)

    new_target_edges = target_edges + deg_target
    new_target_size = target_size + 1

    old_score_part = (source_edges / source_size) + (target_edges / target_size)

    if source_size > 1:
        new_source_edges = source_edges - deg_source
        new_source_size = source_size - 1
        new_score_part = (new_source_edges / new_source_size) + (new_target_edges / new_target_size)
        return new_score_part - old_score_part
    else:
        new_score_part = new_target_edges / new_target_size
        return new_score_part - old_score_part


def apply_move_node(state: PartitionState, v: int, target_cluster: int) -> None:
    """
    Executes a node move and updates the partition state in place.

    Args:
        state (PartitionState): The current state of the partition.
        v (int): The node to be moved.
        target_cluster (int): The index of the destination cluster.

    Raises:
        ValueError: If moving the node would leave a cluster empty or if the
                    target cluster is identical to the source cluster.

    Returns:
        None
    """
    source_cluster = state.cluster_of[v]

    if source_cluster == target_cluster:
        raise ValueError("source and target cluster are identical")

    deg_source = neighbors_in_cluster(state, v, source_cluster)
    deg_target = neighbors_in_cluster(state, v, target_cluster)

    state.internal_edges[source_cluster] -= deg_source
    state.internal_edges[target_cluster] += deg_target

    state.cluster_sizes[source_cluster] -= 1
    state.cluster_sizes[target_cluster] += 1

    state.clusters[source_cluster].remove(v)
    state.clusters[target_cluster].add(v)
    state.cluster_of[v] = target_cluster

    if state.cluster_sizes[source_cluster] == 0:
        delete_cluster(state, source_cluster)


def delete_cluster(state: PartitionState, cluster_idx: int) -> None:
    state.clusters.pop(cluster_idx)
    state.cluster_sizes.pop(cluster_idx)
    state.internal_edges.pop(cluster_idx)

    for node, other_cluster_idx in list(state.cluster_of.items()):
        if other_cluster_idx > cluster_idx:
            state.cluster_of[node] = other_cluster_idx - 1


def best_move_for_node(state: PartitionState, v: int) -> tuple[Optional[int], float]:
    """
    Finds the best target cluster for a given node to maximize density improvement.

    Args:
        state (PartitionState): The current state of the partition.
        v (int): The node to evaluate moves for.

    Returns:
        tuple[Optional[int], float]: A tuple containing the index of the best target
                                     cluster and the expected change in density. If
                                     no improvement is possible, returns (None, 0.0).
    """
    source_cluster = state.cluster_of[v]

    candidate_clusters = {
        state.cluster_of[u] for u in state.G.neighbors(v) if state.cluster_of[u] != source_cluster
    }

    best_cluster: Optional[int] = None
    best_delta = 0.0

    for cluster_idx in candidate_clusters:
        delta = delta_move_node(state, v, cluster_idx)
        if delta > best_delta:
            best_delta = delta
            best_cluster = cluster_idx

    return best_cluster, best_delta
