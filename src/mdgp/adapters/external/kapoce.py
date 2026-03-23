import subprocess
from pathlib import Path

import networkx as nx

Partition = list[set[int]]

def write_pace_instance(G: nx.Graph) -> str:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    lines = [f"p cep {n} {m}"]
    for u,v in sorted(G.edges()):
        lines.append(f"{u+1} {v+1}")

    return "\n".join(lines) + "\n"

def parse_kapoce_edits(output: str) -> list[tuple[int, int]]:
    edits: list[tuple[int, int]] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        u, v = map(int, parts)
        edits.append((u-1, v-1))

    return edits

def apply_edits(G: nx.Graph, edits: list[tuple[int, int]]) -> nx.Graph:
    H = G.copy()
    for u,v in edits:
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        else:
            H.add_edge(u, v)

    return H

def cluster_graph_to_partition(H: nx.Graph) -> Partition:
    return [set(component) for component in nx.connected_components(H)]

def kapoce_partition(G: nx.Graph, executable_path: str | Path) -> Partition:
    executable_path = Path(executable_path)
    input_data = write_pace_instance(G)

    cmd = [str(executable_path)]

    result = subprocess.run(cmd, input=input_data, text=True, capture_output=True, cwd=executable_path.parent)

    edits = parse_kapoce_edits(result.stdout)
    edited_graph = apply_edits(G, edits)

    return cluster_graph_to_partition(edited_graph)