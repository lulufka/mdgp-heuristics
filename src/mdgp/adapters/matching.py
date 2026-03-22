import networkx as nx

Partition = list[set[int]]

def matching_partition(G: nx.Graph) -> Partition:
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
