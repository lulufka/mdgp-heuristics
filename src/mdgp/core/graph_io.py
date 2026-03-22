import json
import networkx as nx
from pathlib import Path


class GraphInstance:
    def __init__(self, name: str, G: nx.Graph):
        self.name = name
        self.G = G

def load_graph_instance(path: str | Path) -> GraphInstance:
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

def load_all_graphs(folder: str | Path):
    folder = Path(folder)
    instances = []

    for file in folder.glob("*.json"):
        instances.append(load_graph_instance(file))

    return instances