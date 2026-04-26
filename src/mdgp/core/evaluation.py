import networkx as nx

from mdgp.core.types import Partition


def is_valid_partition(G: nx.Graph, partition: Partition) -> bool:
    """
    Checks if a given partition is valid for the provided graph.

    A valid partition must cover all nodes in the graph exactly once,
    and none of its clusters can be empty.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): A list of sets of integers representing node clusters.

    Returns:
        bool: True if the partition is valid, False otherwise.
    """
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
    """
    Validates a partition and raises an exception if it is invalid.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): A list of sets of integers representing node clusters.

    Raises:
        ValueError: If the partition is invalid.

    Returns:
        None
    """
    if not is_valid_partition(G, partition):
        raise ValueError(f"Invalid partition: {partition}")


def cluster_density(G: nx.Graph, cluster: set[int]) -> float:
    """
    Calculates the density of a single cluster in the graph.

    The density is defined as the number of edges in the subgraph induced
    by the cluster, divided by the number of nodes in the cluster.

    Args:
        G (nx.Graph): The networkx graph.
        cluster (set[int]): A set of node indices representing a cluster.

    Returns:
        float: The density of the cluster.
    """
    subgraphs = G.subgraph(cluster)
    return subgraphs.number_of_edges() / len(cluster)


def partition_density(G: nx.Graph, partition: Partition) -> float:
    """
    Calculates the total density of a partition.

    The partition density is the sum of the densities of all its clusters.

    Args:
        G (nx.Graph): The networkx graph.
        partition (Partition): A list of sets of integers representing node clusters.

    Returns:
        float: The total density of the partition.
    """
    validate_partition(G, partition)
    return sum(cluster_density(G, cluster) for cluster in partition)


def cluster_edge_count(G: nx.Graph, cluster: set[int]) -> int:
    """
    Counts the number of internal edges in a cluster.

    Args:
        G (nx.Graph): The networkx graph.
        cluster (set[int]): A set of node indices representing a cluster.

    Returns:
        int: The number of edges within the cluster.
    """
    return G.subgraph(cluster).number_of_edges()


def partition_num_clusters(partition: Partition) -> int:
    """
    Gets the number of clusters in a partition.

    Args:
        partition (Partition): A list of sets of integers representing node clusters.

    Returns:
        int: The total number of clusters.
    """
    return len(partition)


def partition_cluster_sizes(partition: Partition) -> list[int]:
    """
    Retrieves the sizes of all clusters in a partition, sorted in descending order.

    Args:
        partition (Partition): A list of sets of integers representing node clusters.

    Returns:
        list[int]: A list of cluster sizes sorted from largest to smallest.
    """
    return sorted((len(cluster) for cluster in partition), reverse=True)
