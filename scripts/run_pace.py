from mdgp.adapters.densest_subgraph import greedy_partition
from mdgp.adapters.external.kapoce import kapoce_partition
from mdgp.adapters.external.leiden import leiden_modularity_partition
from mdgp.adapters.matching import matching_partition
from mdgp.config import KAPOCE_EXECUTABLE, KAPOCE_CONFIG
from mdgp.core.evaluation import partition_density, partition_num_clusters, partition_cluster_sizes
from mdgp.core.graph_io import load_pace_graph_instance
from pathlib import Path
import pandas as pd
import time

from functools import partial

kapoce_heuristic = partial(
    kapoce_partition,
    executable_path=KAPOCE_EXECUTABLE,
    config_path=KAPOCE_CONFIG,
)

folder = Path("data/pace_ce/heur_small")
instance_files = sorted(folder.glob("*.gr"))

results = []

algorithms = [
    ("matching",matching_partition),
    ("greedy", greedy_partition),
    ("leiden modularity", leiden_modularity_partition),
    ("kapoce", kapoce_heuristic),
]

for file_path in instance_files:
    inst = load_pace_graph_instance(file_path)

    G = inst.G
    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"[{inst.name}] Start processing (n={n}, m={m})")

    for algorithm_name, algorithm in algorithms:
        start_time = time.time()
        partition = algorithm(G)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"  -> {algorithm_name} finished in {elapsed_time:.4f}s")

        result = {
            "instance": inst.name,
            "algorithm": algorithm_name,
            "n": n,
            "m": m,
            "time": elapsed_time,
            "density": partition_density(G, partition),
            "num_clusters": partition_num_clusters(partition),
            "cluster_sizes": partition_cluster_sizes(partition),
        }

        results.append(result)

df = pd.DataFrame(results)

pivot_density = df.pivot(index="instance", columns="algorithm", values="density")
pivot_density = pivot_density.sort_index().round(2)
pivot_density.to_csv("results/pace_density_table.csv")

pivot_time = df.pivot(index="instance", columns="algorithm", values="time")
pivot_time = pivot_time.sort_index().round(4)
pivot_time.to_csv("results/pace_time_table.csv")
