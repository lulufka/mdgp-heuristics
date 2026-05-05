from collections.abc import Callable

import networkx as nx

from mdgp.adapters.matching import matching_partition
from mdgp.core.types import Partition


InitialPartitioner = Callable[[nx.Graph], Partition]


def singleton_partition(G: nx.Graph) -> Partition:
    return [{node} for node in G.nodes()]


def all_in_one_partition(G: nx.Graph) -> Partition:
    nodes = set(G.nodes())
    return [nodes] if nodes else []


INITIAL_PARTITIONERS: dict[str, InitialPartitioner] = {
    "matching": matching_partition,
    "singleton": singleton_partition,
    "all_in_one": all_in_one_partition,
}


def get_initial_partitioner(name: str) -> InitialPartitioner:
    try:
        return INITIAL_PARTITIONERS[name]
    except KeyError:
        known = ", ".join(sorted(INITIAL_PARTITIONERS))
        raise ValueError(f"Unknown start partition '{name}'. Known starts: {known}") from None
