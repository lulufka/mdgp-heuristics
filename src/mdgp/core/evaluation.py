import networkx as nx

Partition = list[set[int]]

def is_valid_partition(G: nx.Graph, partition: Partition) -> bool:
    nodes = set(G.nodes())
    seen = set()

    for cluster in partition:
        if len(cluster) == 0:
            return False

        if not cluster.issubset(nodes):
            return False

        if seen & cluster:
            return False

        seen.update(cluster)

    return seen == nodes

def validate_partition(G: nx.Graph, partition: Partition) -> None:
    if not is_valid_partition(G, partition):
        raise ValueError(f"Invalid partition: {partition}")

def cluster_density(G: nx.Graph, cluster: set[int]) -> float:
    subgraphs = G.subgraph(cluster)
    return subgraphs.number_of_edges() / len(cluster)

def partition_density(G: nx.Graph, partition: Partition) -> float:
    validate_partition(G, partition)
    return sum(cluster_density(G, cluster) for cluster in partition)

def cluster_edge_count(G: nx.Graph, cluster: set[int]) -> int:
    return G.subgraph(cluster).number_of_edges()

def partition_num_clusters(partition: Partition) -> int:
    return len(partition)

def partition_cluster_sizes(partition: Partition) -> list[int]:
    return sorted((len(cluster) for cluster in partition), reverse=True)