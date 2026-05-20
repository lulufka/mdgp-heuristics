from collections.abc import Callable
import random

import networkx as nx
from networkx.algorithms.approximation.density import densest_subgraph

from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.matching import matching_partition, maximum_matching_partition, randomized_greedy_matching_partition, \
    low_degree_first_matching_partition, high_degree_first_matching_partition
from mdgp.core.types import Partition

InitialPartitioner = Callable[[nx.Graph], Partition]


def singleton_partition(G: nx.Graph) -> Partition:
    return [{node} for node in G.nodes()]


def all_in_one_partition(G: nx.Graph) -> Partition:
    nodes = set(G.nodes())
    return [nodes] if nodes else []


def random_matching_partition(
    G: nx.Graph,
    random_seed: int | None = None,
) -> Partition:
    rng = random.Random(random_seed)
    edges = list(G.edges())
    rng.shuffle(edges)

    partition: Partition = []
    used = set()

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


INITIAL_PARTITIONERS: dict[str, InitialPartitioner] = {
    "all_in_one": all_in_one_partition,
    "matching": matching_partition,
    "maximum_matching": maximum_matching_partition,
    "random_matching": randomized_greedy_matching_partition,
    "low_degree_matching": low_degree_first_matching_partition,
    "high_degree_matching": high_degree_first_matching_partition,
    "singleton": singleton_partition,
}


def get_initial_partitioner(name: str) -> InitialPartitioner:
    try:
        return INITIAL_PARTITIONERS[name]
    except KeyError:
        known = ", ".join(sorted(INITIAL_PARTITIONERS))
        raise ValueError(f"Unknown start partition '{name}'. Known starts: {known}") from None
