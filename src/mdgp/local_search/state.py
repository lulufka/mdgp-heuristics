from dataclasses import dataclass

import networkx as nx

from mdgp.core.types import Partition


@dataclass
class PartitionState:
    """
    Represents the internal state of a partition for efficient local search updates.
    """
    G: nx.Graph
    clusters: list[set[int]]
    cluster_of: dict[int, int]
    cluster_sizes: list[int]
    internal_edges: list[int]

    def score(self) -> float:
        """
        Calculates the total density score of the current partition.

        Returns:
            float: The sum of densities for all clusters in the partition.
        """
        total = 0.0
        for e, s in zip(self.internal_edges, self.cluster_sizes):
            total += e / s
        return total


def build_partition_state(G: nx.Graph, partition: Partition) -> PartitionState:
    """
    Constructs a fast-update PartitionState from a networkx graph and a partition.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): The initial partition as a list of sets.

    Returns:
        PartitionState: A data structure keeping track of cluster sizes, edge counts,
                        and membership mapping for efficient O(1) lookups.
    """
    clusters = []
    cluster_of = {}

    for idx, cluster in enumerate(partition):
        cluster_copy = set(cluster)
        clusters.append(cluster_copy)
        for v in cluster_copy:
            cluster_of[v] = idx

    cluster_sizes = [len(cluster) for cluster in clusters]

    internal_edges = []
    for cluster in clusters:
        subgraph = G.subgraph(cluster)
        internal_edges.append(subgraph.number_of_edges())

    return PartitionState(G, clusters, cluster_of, cluster_sizes, internal_edges)


def neighbors_in_cluster(state: PartitionState, v: int, cluster_idx: int) -> int:
    """
    Counts how many neighbors of node `v` reside in a specific cluster.

    Args:
        state (PartitionState): The current partition state.
        v (int): The node to query neighbors for.
        cluster_idx (int): The index of the target cluster.

    Returns:
        int: The number of neighbors of `v` inside the target cluster.
    """
    count = 0
    for u in state.G.neighbors(v):
        if state.cluster_of.get(u) == cluster_idx:
            count += 1
    return count
