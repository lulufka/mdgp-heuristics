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

import networkx as nx

from mdgp.core.types import Partition


def maximum_matching_partition(G: nx.Graph) -> Partition:
    matching = nx.max_weight_matching(G, maxcardinality=True)

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

import random
import networkx as nx

from mdgp.core.types import Partition


def randomized_greedy_matching_partition(
        G: nx.Graph,
        random_seed: int | None = None,
) -> Partition:
    rng = random.Random(random_seed)

    edges = list(G.edges())
    rng.shuffle(edges)

    used = set()
    partition: Partition = []

    for u, v in edges:
        if u in used or v in used:
            continue

        partition.append({u, v})
        used.add(u)
        used.add(v)

    for u in G.nodes():
        if u not in used:
            partition.append({u})

    return partition

def low_degree_first_matching_partition(
        G: nx.Graph,
) -> Partition:
    edges = list(G.edges())

    edges.sort(
        key=lambda e: (
            min(G.degree(e[0]), G.degree(e[1])),
            max(G.degree(e[0]), G.degree(e[1])),
        )
    )

    used = set()
    partition: Partition = []

    for u, v in edges:
        if u in used or v in used:
            continue

        partition.append({u, v})
        used.add(u)
        used.add(v)

    for u in G.nodes():
        if u not in used:
            partition.append({u})

    return partition

def high_degree_first_matching_partition(
        G: nx.Graph,
) -> Partition:
    edges = list(G.edges())

    edges.sort(
        key=lambda e: (
            max(G.degree(e[0]), G.degree(e[1])),
            min(G.degree(e[0]), G.degree(e[1])),
        ),
        reverse=True,
    )

    used = set()
    partition: Partition = []

    for u, v in edges:
        if u in used or v in used:
            continue

        partition.append({u, v})
        used.add(u)
        used.add(v)

    for u in G.nodes():
        if u not in used:
            partition.append({u})

    return partition
