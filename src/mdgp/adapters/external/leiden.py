from collections import defaultdict

import igraph
import leidenalg
import networkx as nx

Partition = list[set[int]]

def nx_to_igraph(G: nx.Graph) -> tuple[igraph.Graph, list[int]]:
    node_order = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}

    ig_graph = igraph.Graph()
    ig_graph.add_vertices(len(node_order))
    ig_graph.add_edges([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])

    return ig_graph, node_order

def membership_to_partition(membership: list[int], node_order: list[int]) -> Partition:
    clusters: dict[int, set[int]] = defaultdict(set)

    for vertex_idx, cluster_id in enumerate(membership):
        original_node = node_order[vertex_idx]
        clusters[cluster_id].add(original_node)

    return list(clusters.values())

def leiden_modularity_partition(G: nx.Graph, seed: int | None = None) -> Partition:
    if G.number_of_nodes() == 0:
        return []

    ig_graph, node_order = nx_to_igraph(G)

    if seed is not None:
        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(seed)
        partition = leidenalg.ModularityVertexPartition(ig_graph)
        optimiser.optimise_partition(partition)
    else:
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    return membership_to_partition(partition.membership, node_order)