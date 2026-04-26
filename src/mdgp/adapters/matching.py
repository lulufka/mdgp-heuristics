import networkx as nx

from mdgp.core.types import Partition


def matching_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition of the graph based on a maximal matching.

    Edges in the maximal matching form clusters of size 2. Nodes that
    are not part of the matching are placed into clusters of size 1.

    Args:
        G (nx.Graph): The networkx graph to partition.

    Returns:
        Partition: A partition of the graph nodes based on the maximal matching.
    """
    matching = nx.maximal_matching(G)

    partition: Partition = []
    used = set()

    for u, v in matching:
        partition.append({u, v})
        used.add(u)
        used.add(v)

    for u in G.nodes():
        if u not in used:
            partition.append({u})

    return partition
