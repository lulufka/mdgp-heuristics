from collections.abc import Callable

import networkx as nx
from networkx.algorithms.approximation.density import densest_subgraph

from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.matching import matching_partition
from mdgp.core.types import Partition

InitialPartitioner = Callable[[nx.Graph], Partition]


def singleton_partition(G: nx.Graph) -> Partition:
    return [{node} for node in G.nodes()]


def all_in_one_partition(G: nx.Graph) -> Partition:
    nodes = set(G.nodes())
    return [nodes] if nodes else []


INITIAL_PARTITIONERS: dict[str, InitialPartitioner] = {
    "all_in_one": all_in_one_partition,
    "matching": matching_partition,
    "singleton": singleton_partition,
}


def get_initial_partitioner(name: str) -> InitialPartitioner:
    try:
        return INITIAL_PARTITIONERS[name]
    except KeyError:
        known = ", ".join(sorted(INITIAL_PARTITIONERS))
        raise ValueError(f"Unknown start partition '{name}'. Known starts: {known}") from None
