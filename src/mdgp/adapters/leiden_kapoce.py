from functools import partial

import leidenalg
import networkx as nx

from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import nx_to_igraph, membership_to_partition
from mdgp.config import KAPOCE_EXECUTABLE, KAPOCE_CONFIG
from mdgp.core.types import Partition

kapoce_heuristic = partial(
    kapoce_partition,
    executable_path=KAPOCE_EXECUTABLE,
    config_path=KAPOCE_CONFIG,
)


def partition_to_membership(partition: Partition, node_order: list[int]) -> list[int]:
    """
    Converts a partition format (list of sets) to a membership list for igraph/leidenalg.

    Args:
        partition (Partition): The partition to convert.
        node_order (list[int]): A list mapping the graph nodes to integer indices.

    Returns:
        list[int]: A membership list where the value at index i corresponds to the
                   cluster ID of the node node_order[i].
    """
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    membership = [-1] * len(node_order)

    for cluster_id, cluster in enumerate(partition):
        for node in cluster:
            idx = node_to_idx[node]
            membership[idx] = cluster_id

    return membership


def leiden_mdgp_kapoce_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition of the graph using the Leiden algorithm initialized with Kapoce.

    It first generates an initial partition using the external Kapoce heuristic. Then,
    it refines this partition by applying the Leiden algorithm configured for the
    MDGP objective.

    Args:
        G (nx.Graph): The networkx graph to partition.

    Returns:
        Partition: The refined partition after applying the Leiden algorithm.
    """
    if G.number_of_nodes() == 0:
        return []

    initial_partition = kapoce_heuristic(G)

    ig_graph, node_order = nx_to_igraph(G)
    initial_membership = partition_to_membership(initial_partition, node_order)

    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(42)

    partition = leidenalg.MDGPVertexPartition(ig_graph)
    partition.set_membership(initial_membership)

    max_rounds = 20
    for _ in range(max_rounds):
        diff = optimiser.move_nodes(partition)
        if diff == 0:
            break

    return membership_to_partition(partition.membership, node_order)
