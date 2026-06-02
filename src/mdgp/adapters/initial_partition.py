from collections.abc import Callable
from itertools import combinations
import random
from typing import Optional

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


def clique_seed_partition(
    G: nx.Graph,
    *,
    use_diamonds: bool = True,
    use_triangles: bool = True,
    use_matching_fallback: bool = True,
    random_seed: Optional[int] = None,
) -> Partition:
    """
    Builds a start partition from dense small subgraphs.

    Priority:
    1. disjoint diamonds / K4s
    2. disjoint triangles
    3. matching edges
    4. singleton nodes

    A diamond is a 4-node subgraph with at least 5 internal edges. K4 is accepted too.
    """
    rng = random.Random(random_seed)
    unused = set(G.nodes())
    partition: Partition = []

    degrees = dict(G.degree())

    if use_diamonds:
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        nodes.sort(key=lambda v: degrees[v], reverse=True)

        diamond_candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []

        for u in nodes:
            nbrs = [v for v in G.neighbors(u) if v != u]
            if len(nbrs) < 3:
                continue

            nbrs.sort(key=lambda v: degrees[v], reverse=True)
            nbrs = nbrs[:80]

            for a, b, c in combinations(nbrs, 3):
                quad = (u, a, b, c)
                edge_count = _induced_edge_count(G, quad)

                if edge_count >= 5:
                    score = sum(degrees[x] for x in quad)
                    diamond_candidates.append((edge_count, score, quad))

        diamond_candidates.sort(reverse=True)

        for _, _, quad in diamond_candidates:
            q = set(quad)
            if q <= unused:
                partition.append(q)
                unused -= q

    if use_triangles:
        triangle_candidates: list[tuple[int, tuple[int, int, int]]] = []

        for u in list(unused):
            nbrs = [v for v in G.neighbors(u) if v in unused and v != u]
            nbrs.sort(key=lambda v: degrees[v], reverse=True)
            nbrs = nbrs[:120]

            for a, b in combinations(nbrs, 2):
                if a in unused and b in unused and G.has_edge(a, b):
                    tri = (u, a, b)
                    score = sum(degrees[x] for x in tri)
                    triangle_candidates.append((score, tri))

        triangle_candidates.sort(reverse=True)

        for _, tri in triangle_candidates:
            t = set(tri)
            if t <= unused:
                partition.append(t)
                unused -= t

    if use_matching_fallback and unused:
        H = G.subgraph(unused)
        matching = nx.maximal_matching(H)

        for u, v in matching:
            if u in unused and v in unused:
                partition.append({u, v})
                unused.remove(u)
                unused.remove(v)

    for v in unused:
        partition.append({v})

    return partition


def _induced_edge_count(G: nx.Graph, nodes: tuple[int, ...]) -> int:
    count = 0
    for u, v in combinations(nodes, 2):
        if G.has_edge(u, v):
            count += 1
    return count


INITIAL_PARTITIONERS: dict[str, InitialPartitioner] = {
    "all_in_one": all_in_one_partition,
    "clique_seed": clique_seed_partition,
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
