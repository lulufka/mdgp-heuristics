from collections import defaultdict

import igraph
import leidenalg
import networkx as nx

from mdgp.core.types import Partition


def nx_to_igraph(G: nx.Graph) -> tuple[igraph.Graph, list[int]]:
    """
    Converts a networkx graph into an igraph graph.

    Args:
        G (nx.Graph): The networkx graph to convert.

    Returns:
        tuple[igraph.Graph, list[int]]: A tuple containing the igraph graph and a list
                                        mapping igraph vertex indices to original node IDs.
    """
    node_order = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    ig_graph = igraph.Graph()
    ig_graph.add_vertices(len(node_order))
    ig_graph.add_edges([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])

    return ig_graph, node_order


def membership_to_partition(membership: list[int], node_order: list[int]) -> Partition:
    """
    Converts an igraph/leidenalg membership list back into a Partition format.

    Args:
        membership (list[int]): A list where index corresponds to the vertex index and
                                value corresponds to its cluster ID.
        node_order (list[int]): A list mapping vertex indices to the original node IDs.

    Returns:
        Partition: A partition of the graph nodes into disjoint sets.
    """
    clusters: dict[int, set[int]] = defaultdict(set)

    for vertex_idx, cluster_id in enumerate(membership):
        original_node = node_order[vertex_idx]
        clusters[cluster_id].add(original_node)

    return list(clusters.values())


def leiden_modularity_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition using the Leiden algorithm with the standard Modularity objective.

    Args:
        G (nx.Graph): The networkx graph to partition.

    Returns:
        Partition: A partition of the graph.
    """
    if G.number_of_nodes() == 0:
        return []

    ig_graph, node_order = nx_to_igraph(G)

    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    return membership_to_partition(partition.membership, node_order)


def leiden_mdgp_partition(G: nx.Graph) -> Partition:
    """
    Computes a partition using the Leiden algorithm with the MDGP objective.

    Args:
        G (nx.Graph): The networkx graph to partition.

    Returns:
        Partition: A partition of the graph optimized for MDGP.
    """
    if G.number_of_nodes() == 0:
        return []

    ig_graph, node_order = nx_to_igraph(G)
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(42)

    partition = leidenalg.MDGPVertexPartition(ig_graph)

    max_rounds = 20
    for _ in range(max_rounds):
        diff = optimiser.move_nodes(partition)
        if diff == 0:
            break

    return membership_to_partition(partition.membership, node_order)
