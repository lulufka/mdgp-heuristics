import networkx as nx
from networkx.algorithms.approximation import densest_subgraph

from mdgp.core.types import Partition


def greedy_partition(G: nx.Graph) -> Partition:
    """
    Greedily partitions a graph by iteratively extracting the densest subgraph.

    At each step, the densest subgraph approximation is found and extracted
    as a new cluster. This process repeats until no nodes remain. Any remaining
    isolated nodes (if the densest subgraph is empty) are placed in their own
    individual clusters of size 1.

    Args:
        G (nx.Graph): The networkx graph to partition.

    Returns:
        Partition: A partition constructed by iteratively finding the densest subgraph.
    """
    remaining = set(G.nodes())
    partition: Partition = []

    while remaining:
        H = G.subgraph(remaining).copy()
        _, cluster = densest_subgraph(H)

        if not cluster:
            for v in remaining:
                partition.append({v})
            break

        partition.append(set(cluster))
        remaining -= set(cluster)

    return partition
