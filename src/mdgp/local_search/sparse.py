import random
from itertools import combinations
from typing import Optional

import networkx as nx

from mdgp.local_search.merge import apply_merge_clusters, delta_merge_clusters, neighboring_cluster_pairs
from mdgp.local_search.move import best_move_for_node, apply_move_node
from mdgp.local_search.state import PartitionState, build_partition_state, neighbors_in_cluster


def _cluster_score(state: PartitionState, cluster_idx: int) -> float:
    return state.internal_edges[cluster_idx] / state.cluster_sizes[cluster_idx]


def _replace_state(state: PartitionState, clusters: list[set[int]]) -> None:
    new_state = build_partition_state(state.G, [cluster for cluster in clusters if cluster])
    state.clusters = new_state.clusters
    state.cluster_of = new_state.cluster_of
    state.cluster_sizes = new_state.cluster_sizes
    state.internal_edges = new_state.internal_edges


def _edges_inside_nodes(state: PartitionState, nodes: set[int]) -> int:
    return state.G.subgraph(nodes).number_of_edges()


def _edges_from_nodes_to_cluster(
    state: PartitionState,
    nodes: set[int],
    cluster_idx: int,
    *,
    exclude: set[int] | None = None,
) -> int:
    exclude = exclude or set()
    target_nodes = state.clusters[cluster_idx] - exclude
    count = 0

    for v in nodes:
        for u in state.G.neighbors(v):
            if u in target_nodes:
                count += 1

    return count


def _add_node_delta(state: PartitionState, v: int, target_cluster: int) -> float:
    size = state.cluster_sizes[target_cluster]
    edges = state.internal_edges[target_cluster]
    deg_target = neighbors_in_cluster(state, v, target_cluster)

    return ((edges + deg_target) / (size + 1)) - (edges / size)


def _apply_add_node_to_cluster(state: PartitionState, v: int, target_cluster: int) -> None:
    deg_target = neighbors_in_cluster(state, v, target_cluster)
    state.clusters[target_cluster].add(v)
    state.cluster_of[v] = target_cluster
    state.cluster_sizes[target_cluster] += 1
    state.internal_edges[target_cluster] += deg_target


def _apply_add_node_as_singleton(state: PartitionState, v: int) -> None:
    state.clusters.append({v})
    state.cluster_of[v] = len(state.clusters) - 1
    state.cluster_sizes.append(1)
    state.internal_edges.append(0)


def best_small_cluster_move(
    state: PartitionState,
    *,
    max_cluster_size: int = 5,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[int, int]], float]:
    pairs = neighboring_cluster_pairs(state)
    if random_seed is not None:
        random.Random(random_seed).shuffle(pairs)

    best_pair: Optional[tuple[int, int]] = None
    best_delta = 0.0

    for a, b in pairs:
        candidate_pairs = []
        if state.cluster_sizes[a] <= max_cluster_size:
            candidate_pairs.append((a, b))
        if state.cluster_sizes[b] <= max_cluster_size:
            candidate_pairs.append((b, a))

        for source, target in candidate_pairs:
            delta = delta_merge_clusters(state, source, target)
            if delta > best_delta:
                best_pair = (source, target)
                best_delta = delta

    return best_pair, best_delta


def apply_small_cluster_move(state: PartitionState, source: int, target: int) -> None:
    apply_merge_clusters(state, source, target)


def delta_move_node_set(state: PartitionState, nodes: set[int], target_cluster: int) -> float:
    if not nodes:
        return float("-inf")

    source_cluster = state.cluster_of[next(iter(nodes))]
    if target_cluster == source_cluster:
        return float("-inf")
    if any(state.cluster_of[v] != source_cluster for v in nodes):
        return float("-inf")

    source_size = state.cluster_sizes[source_cluster]
    target_size = state.cluster_sizes[target_cluster]
    if len(nodes) > source_size:
        return float("-inf")

    source_edges = state.internal_edges[source_cluster]
    target_edges = state.internal_edges[target_cluster]
    moving_internal_edges = _edges_inside_nodes(state, nodes)
    moving_to_source_edges = _edges_from_nodes_to_cluster(
        state,
        nodes,
        source_cluster,
        exclude=nodes,
    )
    moving_to_target_edges = _edges_from_nodes_to_cluster(state, nodes, target_cluster)

    old_score = (source_edges / source_size) + (target_edges / target_size)
    new_target_score = (
        target_edges + moving_internal_edges + moving_to_target_edges
    ) / (target_size + len(nodes))

    if source_size == len(nodes):
        new_score = new_target_score
    else:
        new_source_score = (
            source_edges - moving_internal_edges - moving_to_source_edges
        ) / (source_size - len(nodes))
        new_score = new_source_score + new_target_score

    return new_score - old_score


def apply_move_node_set(state: PartitionState, nodes: set[int], target_cluster: int) -> None:
    source_cluster = state.cluster_of[next(iter(nodes))]
    if target_cluster == source_cluster:
        raise ValueError("source and target cluster are identical")
    if any(state.cluster_of[v] != source_cluster for v in nodes):
        raise ValueError("all moved nodes must come from the same cluster")

    clusters = [set(cluster) for cluster in state.clusters]
    clusters[source_cluster].difference_update(nodes)
    clusters[target_cluster].update(nodes)
    _replace_state(state, clusters)


def best_pair_move(
    state: PartitionState,
    *,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[set[int], int]], float]:
    edges = list(state.G.edges())
    if random_seed is not None:
        random.Random(random_seed).shuffle(edges)

    best: Optional[tuple[set[int], int]] = None
    best_delta = 0.0

    for u, v in edges:
        source_cluster = state.cluster_of[u]
        if state.cluster_of[v] != source_cluster:
            continue

        nodes = {u, v}
        candidate_clusters = {
            state.cluster_of[w]
            for node in nodes
            for w in state.G.neighbors(node)
            if state.cluster_of[w] != source_cluster
        }

        for target_cluster in candidate_clusters:
            delta = delta_move_node_set(state, nodes, target_cluster)
            if delta > best_delta:
                best = (nodes, target_cluster)
                best_delta = delta

    return best, best_delta


def _dissolve_candidate(
    state: PartitionState,
    cluster_indices: set[int],
) -> tuple[list[set[int]], float]:
    removed_nodes = set().union(*(state.clusters[i] for i in cluster_indices))
    base_clusters = [
        set(cluster)
        for idx, cluster in enumerate(state.clusters)
        if idx not in cluster_indices
    ]

    if not base_clusters:
        return [], float("-inf")

    temp_state = build_partition_state(state.G, base_clusters)
    nodes = sorted(removed_nodes, key=lambda node: state.G.degree[node], reverse=True)

    for v in nodes:
        candidate_clusters = {
            temp_state.cluster_of[u]
            for u in state.G.neighbors(v)
            if u in temp_state.cluster_of
        }

        best_target = None
        best_delta = 0.0
        for target_cluster in candidate_clusters:
            delta = _add_node_delta(temp_state, v, target_cluster)
            if delta > best_delta:
                best_delta = delta
                best_target = target_cluster

        if best_target is None:
            _apply_add_node_as_singleton(temp_state, v)
        else:
            _apply_add_node_to_cluster(temp_state, v, best_target)

    old_score = state.score()
    new_score = temp_state.score()
    return [set(cluster) for cluster in temp_state.clusters], new_score - old_score


def best_small_cluster_dissolve(
    state: PartitionState,
    *,
    max_cluster_size: int = 3,
    random_seed: int | None = None,
) -> tuple[Optional[list[set[int]]], float]:
    cluster_indices = [
        idx
        for idx, size in enumerate(state.cluster_sizes)
        if 1 < size <= max_cluster_size
    ]
    if random_seed is not None:
        random.Random(random_seed).shuffle(cluster_indices)

    best_clusters: Optional[list[set[int]]] = None
    best_delta = 0.0

    for cluster_idx in cluster_indices:
        clusters, delta = _dissolve_candidate(state, {cluster_idx})
        if clusters and delta > best_delta:
            best_clusters = clusters
            best_delta = delta

    return best_clusters, best_delta


def apply_rebuilt_clusters(state: PartitionState, clusters: list[set[int]]) -> None:
    _replace_state(state, clusters)


def _score_clusters(state: PartitionState, clusters: list[set[int]]) -> float:
    return sum(_edges_inside_nodes(state, cluster) / len(cluster) for cluster in clusters)


def _is_connected_mask(mask: int, neighbor_masks: list[int]) -> bool:
    if mask & (mask - 1) == 0:
        return True

    start = mask & -mask
    seen = start
    frontier = start

    while frontier:
        bit = frontier & -frontier
        frontier ^= bit
        idx = bit.bit_length() - 1
        unseen_neighbors = neighbor_masks[idx] & mask & ~seen
        seen |= unseen_neighbors
        frontier |= unseen_neighbors

    return seen == mask


def _optimal_small_partition(
    state: PartitionState,
    nodes: set[int],
    *,
    max_cluster_size: int,
) -> tuple[list[set[int]], float]:
    ordered_nodes = list(nodes)
    n = len(ordered_nodes)
    full_mask = (1 << n) - 1
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}

    neighbor_masks = [0] * n
    for idx, node in enumerate(ordered_nodes):
        mask = 0
        for neighbor in state.G.neighbors(node):
            neighbor_idx = node_to_idx.get(neighbor)
            if neighbor_idx is not None:
                mask |= 1 << neighbor_idx
        neighbor_masks[idx] = mask

    edge_counts = [0] * (1 << n)
    sizes = [0] * (1 << n)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        idx = bit.bit_length() - 1
        rest = mask ^ bit
        sizes[mask] = sizes[rest] + 1
        edge_counts[mask] = edge_counts[rest] + (neighbor_masks[idx] & rest).bit_count()

    cluster_scores: list[float | None] = [None] * (1 << n)
    for mask in range(1, 1 << n):
        if sizes[mask] > max_cluster_size:
            continue
        if not _is_connected_mask(mask, neighbor_masks):
            continue
        cluster_scores[mask] = edge_counts[mask] / sizes[mask]

    dp = [float("-inf")] * (1 << n)
    cluster_counts = [0] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0.0

    for mask in range(1, 1 << n):
        first_bit = mask & -mask
        submask = mask
        while submask:
            if submask & first_bit and cluster_scores[submask] is not None:
                rest = mask ^ submask
                candidate_score = cluster_scores[submask] + dp[rest]
                candidate_count = 1 + cluster_counts[rest]
                if (
                    candidate_score > dp[mask] + 1e-12
                    or (
                        abs(candidate_score - dp[mask]) <= 1e-12
                        and candidate_count > cluster_counts[mask]
                    )
                ):
                    dp[mask] = candidate_score
                    cluster_counts[mask] = candidate_count
                    choice[mask] = submask
            submask = (submask - 1) & mask

    clusters = []
    mask = full_mask
    while mask:
        selected = choice[mask]
        clusters.append(
            {
                ordered_nodes[idx]
                for idx in range(n)
                if selected & (1 << idx)
            }
        )
        mask ^= selected

    return clusters, dp[full_mask]


def best_exact_pair_repack(
    state: PartitionState,
    *,
    max_nodes: int = 12,
    max_cluster_size: int = 6,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[set[int], list[set[int]]]], float]:
    pairs = neighboring_cluster_pairs(state)
    if random_seed is not None:
        random.Random(random_seed).shuffle(pairs)

    best: Optional[tuple[set[int], list[set[int]]]] = None
    best_delta = 0.0

    for a, b in pairs:
        nodes = state.clusters[a] | state.clusters[b]
        if len(nodes) > max_nodes:
            continue

        old_clusters = [state.clusters[a], state.clusters[b]]
        old_score = _score_clusters(state, old_clusters)
        new_clusters, new_score = _optimal_small_partition(
            state,
            nodes,
            max_cluster_size=max_cluster_size,
        )
        delta = new_score - old_score

        if delta > best_delta:
            best = ({a, b}, new_clusters)
            best_delta = delta

    return best, best_delta


def _neighboring_cluster_sets(
    state: PartitionState,
    *,
    group_size: int,
    max_nodes: int,
    max_neighbors_per_cluster: int,
    max_groups: int,
    random_seed: int | None = None,
) -> list[set[int]]:
    neighbors: dict[int, set[int]] = {idx: set() for idx in range(len(state.clusters))}
    for a, b in neighboring_cluster_pairs(state):
        neighbors[a].add(b)
        neighbors[b].add(a)

    rng = random.Random(random_seed)
    groups: set[tuple[int, ...]] = set()

    for center, center_neighbors in neighbors.items():
        candidate_neighbors = list(center_neighbors)
        if random_seed is not None:
            rng.shuffle(candidate_neighbors)
        candidate_neighbors.sort(
            key=lambda idx: state.cluster_sizes[idx],
        )
        candidate_neighbors = candidate_neighbors[:max_neighbors_per_cluster]

        for selected in combinations(candidate_neighbors, group_size - 1):
            group = {center, *selected}
            node_count = sum(state.cluster_sizes[idx] for idx in group)
            if node_count <= max_nodes:
                groups.add(tuple(sorted(group)))

    result = [set(group) for group in groups]
    if random_seed is not None:
        rng.shuffle(result)
    else:
        result.sort(key=lambda group: tuple(sorted(group)))

    return result[:max_groups]


def best_exact_multi_repack(
    state: PartitionState,
    *,
    group_size: int = 3,
    max_nodes: int = 12,
    max_cluster_size: int = 6,
    max_neighbors_per_cluster: int = 6,
    max_groups: int = 100,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[set[int], list[set[int]]]], float]:
    groups = _neighboring_cluster_sets(
        state,
        group_size=group_size,
        max_nodes=max_nodes,
        max_neighbors_per_cluster=max_neighbors_per_cluster,
        max_groups=max_groups,
        random_seed=random_seed,
    )

    best: Optional[tuple[set[int], list[set[int]]]] = None
    best_delta = 0.0

    for group in groups:
        nodes = set().union(*(state.clusters[idx] for idx in group))
        old_clusters = [state.clusters[idx] for idx in group]
        old_score = _score_clusters(state, old_clusters)
        new_clusters, new_score = _optimal_small_partition(
            state,
            nodes,
            max_cluster_size=max_cluster_size,
        )
        delta = new_score - old_score

        if delta > best_delta:
            best = (set(group), new_clusters)
            best_delta = delta

    return best, best_delta


def apply_exact_repack(
    state: PartitionState,
    replaced_indices: set[int],
    new_clusters: list[set[int]],
) -> None:
    clusters = [
        set(cluster)
        for idx, cluster in enumerate(state.clusters)
        if idx not in replaced_indices
    ]
    clusters.extend(set(cluster) for cluster in new_clusters if cluster)
    _replace_state(state, clusters)


def delta_split_node_set(state: PartitionState, nodes: set[int]) -> float:
    if not nodes:
        return float("-inf")

    cluster_idx = state.cluster_of[next(iter(nodes))]
    if any(state.cluster_of[v] != cluster_idx for v in nodes):
        return float("-inf")

    cluster = state.clusters[cluster_idx]
    if len(nodes) >= len(cluster):
        return float("-inf")

    complement = cluster - nodes
    old_score = state.internal_edges[cluster_idx] / state.cluster_sizes[cluster_idx]
    new_score = (
        _edges_inside_nodes(state, nodes) / len(nodes)
        + _edges_inside_nodes(state, complement) / len(complement)
    )

    return new_score - old_score


def best_exact_small_split(
    state: PartitionState,
    *,
    max_cluster_size: int = 10,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[int, set[int], set[int]]], float]:
    cluster_indices = [
        idx
        for idx, size in enumerate(state.cluster_sizes)
        if 2 <= size <= max_cluster_size
    ]
    if random_seed is not None:
        random.Random(random_seed).shuffle(cluster_indices)

    best: Optional[tuple[int, set[int], set[int]]] = None
    best_delta = 0.0

    for cluster_idx in cluster_indices:
        nodes = list(state.clusters[cluster_idx])
        if random_seed is not None:
            random.Random(random_seed + cluster_idx).shuffle(nodes)

        # Enumerate one side only: S and C-S are the same split.
        max_subset_size = len(nodes) // 2
        for subset_size in range(1, max_subset_size + 1):
            if subset_size == len(nodes) - subset_size:
                subset_iter = combinations(nodes[1:], subset_size - 1)
                subsets = ({nodes[0], *subset} for subset in subset_iter)
            else:
                subsets = (set(subset) for subset in combinations(nodes, subset_size))

            for a in subsets:
                b = state.clusters[cluster_idx] - a
                delta = delta_split_node_set(state, a)
                if delta > best_delta:
                    best = (cluster_idx, set(a), set(b))
                    best_delta = delta

    return best, best_delta


def best_peel_node(
    state: PartitionState,
    *,
    random_seed: int | None = None,
) -> tuple[Optional[int], float]:
    nodes = list(state.G.nodes())
    if random_seed is not None:
        random.Random(random_seed).shuffle(nodes)

    best_node: Optional[int] = None
    best_delta = 0.0

    for v in nodes:
        cluster_idx = state.cluster_of[v]
        if state.cluster_sizes[cluster_idx] <= 1:
            continue

        delta = delta_peel_node_as_singleton(state, v)
        if delta > best_delta:
            best_node = v
            best_delta = delta

    return best_node, best_delta


def best_ruin_and_recreate(
    state: PartitionState,
    *,
    fraction: float = 0.10,
    min_clusters: int = 1,
    random_seed: int | None = None,
) -> tuple[Optional[list[set[int]]], float]:
    if len(state.clusters) < 2:
        return None, 0.0

    ruin_count = max(min_clusters, int(len(state.clusters) * fraction))
    ruin_count = min(ruin_count, len(state.clusters) - 1)
    if ruin_count <= 0:
        return None, 0.0

    rng = random.Random(random_seed)
    ranked_clusters = list(range(len(state.clusters)))
    rng.shuffle(ranked_clusters)
    ranked_clusters.sort(key=lambda idx: (_cluster_score(state, idx), state.cluster_sizes[idx]))

    ruined = set(ranked_clusters[:ruin_count])
    clusters, delta = _dissolve_candidate(state, ruined)
    if not clusters or delta <= 0:
        return None, 0.0

    return clusters, delta


def bridge_split_candidate(
    state: PartitionState,
    cluster_idx: int,
    *,
    random_seed: int | None = None,
) -> tuple[Optional[set[int]], Optional[set[int]], float]:
    cluster = state.clusters[cluster_idx]

    if len(cluster) < 4:
        return None, None, float("-inf")

    H = state.G.subgraph(cluster).copy()
    if not nx.is_connected(H):
        return None, None, float("-inf")

    bridges = list(nx.bridges(H))
    if random_seed is not None:
        random.Random(random_seed).shuffle(bridges)

    old_score = state.internal_edges[cluster_idx] / state.cluster_sizes[cluster_idx]
    best_a: Optional[set[int]] = None
    best_b: Optional[set[int]] = None
    best_delta = 0.0

    for u, v in bridges:
        H.remove_edge(u, v)
        components = list(nx.connected_components(H))
        H.add_edge(u, v)

        if len(components) != 2:
            continue

        a = set(components[0])
        b = set(components[1])
        edges_a = state.G.subgraph(a).number_of_edges()
        edges_b = state.G.subgraph(b).number_of_edges()
        delta = (edges_a / len(a)) + (edges_b / len(b)) - old_score

        if delta > best_delta:
            best_a = a
            best_b = b
            best_delta = delta

    return best_a, best_b, best_delta


def best_bridge_split(
    state: PartitionState,
    *,
    random_seed: int | None = None,
) -> tuple[Optional[tuple[int, set[int], set[int]]], float]:
    cluster_indices = list(range(len(state.clusters)))
    if random_seed is not None:
        random.Random(random_seed).shuffle(cluster_indices)

    best: Optional[tuple[int, set[int], set[int]]] = None
    best_delta = 0.0

    for cluster_idx in cluster_indices:
        a, b, delta = bridge_split_candidate(
            state,
            cluster_idx,
            random_seed=random_seed,
        )

        if a is None or b is None:
            continue

        if delta > best_delta:
            best = (cluster_idx, a, b)
            best_delta = delta

    return best, best_delta


def delta_peel_node_as_singleton(state: PartitionState, v: int) -> float:
    cluster_idx = state.cluster_of[v]
    size = state.cluster_sizes[cluster_idx]

    if size <= 1:
        return float("-inf")

    internal_edges = state.internal_edges[cluster_idx]
    internal_degree = neighbors_in_cluster(state, v, cluster_idx)

    old_score = internal_edges / size
    new_score = (internal_edges - internal_degree) / (size - 1)

    return new_score - old_score


def apply_peel_node_as_singleton(state: PartitionState, v: int) -> None:
    cluster_idx = state.cluster_of[v]

    if state.cluster_sizes[cluster_idx] <= 1:
        raise ValueError("cannot peel a singleton cluster")

    internal_degree = neighbors_in_cluster(state, v, cluster_idx)

    state.clusters[cluster_idx].remove(v)
    state.cluster_sizes[cluster_idx] -= 1
    state.internal_edges[cluster_idx] -= internal_degree

    state.clusters.append({v})
    state.cluster_sizes.append(1)
    state.internal_edges.append(0)
    state.cluster_of[v] = len(state.clusters) - 1


def best_low_degree_peel(
    state: PartitionState,
    *,
    max_internal_degree: int = 1,
    random_seed: int | None = None,
) -> tuple[Optional[int], float]:
    nodes = list(state.G.nodes())
    if random_seed is not None:
        random.Random(random_seed).shuffle(nodes)

    best_node: Optional[int] = None
    best_delta = 0.0

    for v in nodes:
        cluster_idx = state.cluster_of[v]
        if state.cluster_sizes[cluster_idx] <= 1:
            continue

        internal_degree = neighbors_in_cluster(state, v, cluster_idx)
        if internal_degree > max_internal_degree:
            continue

        delta = delta_peel_node_as_singleton(state, v)
        if delta > best_delta:
            best_node = v
            best_delta = delta

    return best_node, best_delta


def best_low_degree_move(
    state: PartitionState,
    *,
    max_graph_degree: int = 4,
    random_seed: int | None = None,
) -> tuple[Optional[int], Optional[int], float]:
    nodes = [
        node
        for node, degree in state.G.degree()
        if degree <= max_graph_degree
    ]
    if random_seed is not None:
        random.Random(random_seed).shuffle(nodes)

    best_node: Optional[int] = None
    best_target: Optional[int] = None
    best_delta = 0.0

    for v in nodes:
        target_cluster, delta = best_move_for_node(state, v)
        if target_cluster is not None and delta > best_delta:
            best_node = v
            best_target = target_cluster
            best_delta = delta

    return best_node, best_target, best_delta
