from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import leiden_modularity_partition
from mdgp.adapters.matching import matching_partition
from mdgp.core.evaluation import partition_density, partition_num_clusters, partition_cluster_sizes
from mdgp.core.graph_io import load_all_graphs
import pandas as pd

from functools import partial

kapoce_heuristic = partial(
    kapoce_partition,
    executable_path="/Users/Tessa/Uni/Masterarbeit/Repos zum Vergleichen/cluster_editing/build/heuristic"
)

instances = load_all_graphs("data/tiny")

results = []

algorithms = [
    ("matching",matching_partition),
    ("greedy", greedy_partition),
    ("leiden_modularity", leiden_modularity_partition),
    ("kapoce", kapoce_heuristic),
]

for inst in instances:
    G = inst.G

    for algorithm_name, algorithm in algorithms:
        partition = algorithm(G)

        result = {
            "instance": inst.name,
            "algorithm": algorithm_name,
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "density": partition_density(G, partition),
            "num_clusters": partition_num_clusters(partition),
            "cluster_sizes": partition_cluster_sizes(partition),
        }

        results.append(result)


df = pd.DataFrame(results)

pivot = df.pivot(index="instance", columns="algorithm", values="density")
pivot = pivot.sort_index()
pivot = pivot.round(2)

pivot.to_csv("results/tiny_density_table.csv")
print(pivot)