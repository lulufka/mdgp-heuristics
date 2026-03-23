import json
import networkx as nx
from pathlib import Path


class GraphInstance:
    def __init__(self, name: str, G: nx.Graph):
        self.name = name
        self.G = G

def load_json_graph_instance(path: str | Path) -> GraphInstance:
    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    name = data["name"]
    n = data["n"]
    edges = data["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return GraphInstance(name, G)

def load_all_json_graphs(folder: str | Path) -> list[GraphInstance]:
    folder = Path(folder)
    return [load_json_graph_instance(file) for file in folder.glob("*.json")]

def load_pace_graph_instance(path: str | Path) -> GraphInstance:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

        n = None
        edges = []

        for line in lines:
            if line.startswith("p"):
                parts = line.split()

                n = int(parts[2])
                continue

            u, v = map(int, line.split())
            edges.append((u-1, v-1))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return GraphInstance(name=path.stem, G=G)

def load_all_pace_graphs(folder: str | Path) -> list[GraphInstance]:
    folder = Path(folder)
    return [load_pace_graph_instance(file) for file in folder.glob("*.gr")]
