import pandas as pd

from mdgp.core.graph_io import load_instances
from mdgp.adapters.initial_partition import singleton_partition, matching_partition, all_in_one_partition
from mdgp.local_search.search import (
    refine_partition_move_first_improvement,
    refine_partition_move_best_improvement,
    refine_partition_merge_best_improvement,
    refine_partition_merge_first_improvement,
    refine_partition_split_min_cut,
)

instances = load_instances("data/er_graphs/small")

tests = [
    ("matching", "move_first", matching_partition, refine_partition_move_first_improvement),
    ("matching", "move_best", matching_partition, refine_partition_move_best_improvement),
    ("singleton", "merge_first", singleton_partition, refine_partition_merge_first_improvement),
    ("singleton", "merge_best", singleton_partition, refine_partition_merge_best_improvement),
    ("all_in_one", "split_min_cut", all_in_one_partition, refine_partition_split_min_cut),
]

rows = []

for inst in instances:
    G = inst.G

    for start_name, method_name, start_fn, refiner in tests:
        start_partition = start_fn(G)

        result = refiner(
            G,
            start_partition,
            max_passes=10_000,
            max_moves=None,
        )

        rows.append({
            "instance": inst.name,
            "start": start_name,
            "method": method_name,
            "passes_until_stop": result.num_passes,
            "operations_until_stop": result.num_moves,
            "final_score": result.final_score,
        })

df = pd.DataFrame(rows)

print(df.groupby(["start", "method"]).agg(
    mean_passes=("passes_until_stop", "mean"),
    max_passes=("passes_until_stop", "max"),
    mean_ops=("operations_until_stop", "mean"),
    max_ops=("operations_until_stop", "max"),
    mean_score=("final_score", "mean"),
).round(2))