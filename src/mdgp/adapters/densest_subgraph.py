import networkx as nx
from networkx.algorithms.approximation import densest_subgraph

Partition = list[set[int]]

def greedy_partition(G: nx.Graph) ->Partition:
    remaining = set(G.nodes())
    partition: Partition = []

    while remaining:
        H = G.subgraph(remaining).copy()
        _, cluster = densest_subgraph(H)

        partition.append(set(cluster))
        remaining -= set(cluster)

    return partition

