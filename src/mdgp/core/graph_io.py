import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class GraphInstance:
    """
    Represents a graph instance with a name and the graph itself.
    """
    name: str
    G: nx.Graph


def load_json_graph_instance(path: str | Path) -> GraphInstance:
    """
    Loads a graph instance from a JSON file.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        GraphInstance: A dataclass containing the graph name and the parsed NetworkX graph.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data["name"]
    n = data["n"]
    edges = data["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return GraphInstance(name, G)


def load_all_json_graphs(folder: str | Path) -> list[GraphInstance]:
    """
    Loads all JSON graph instances from a given directory.

    Args:
        folder (str | Path): Path to the directory containing .json files.

    Returns:
        list[GraphInstance]: A list of loaded graph instances.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.json"))
    return [load_json_graph_instance(file) for file in files]


def load_pace_graph_instance(path: str | Path) -> GraphInstance:
    """
    Loads a graph instance from a .gr file (used in the PACE challenge).

    Args:
        path (str | Path): Path to the .gr file.

    Returns:
        GraphInstance: A dataclass containing the graph name and the parsed NetworkX graph.
    """
    path = Path(path)

    n = 0
    edges = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("p"):
                parts = line.split()
                n = int(parts[2])
                continue

            u, v = map(int, line.split())
            edges.append((u - 1, v - 1))

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    return GraphInstance(name=path.stem, G=G)


def load_all_pace_graphs(folder: str | Path) -> list[GraphInstance]:
    """
    Loads all PACE graph instances from a given directory.

    Args:
        folder (str | Path): Path to the directory containing .gr files.

    Returns:
        list[GraphInstance]: A list of loaded graph instances.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.gr"))
    return [load_pace_graph_instance(file) for file in files]


def load_snap_graph_instance(path: str | Path) -> GraphInstance:
    """
    Loads a graph instance from an edge list text file (used in SNAP datasets).

    Args:
        path (str | Path): Path to the .txt file.

    Returns:
        GraphInstance: A dataclass containing the graph name and the parsed NetworkX graph.
    """
    path = Path(path)

    G = nx.read_edgelist(path, comments='#', nodetype=int, create_using=nx.Graph)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")

    return GraphInstance(name=path.stem, G=G)


def load_all_snap_graphs(folder: str | Path) -> list[GraphInstance]:
    """
    Loads all SNAP graph instances from a given directory.

    Args:
        folder (str | Path): Path to the directory containing .txt files.

    Returns:
        list[GraphInstance]: A list of loaded graph instances.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    return [load_snap_graph_instance(file) for file in files]


def infer_dataset_type(folder: str | Path) -> str:
    """
    Infers the dataset type (json, pace, or snap) based on the files in a directory.

    Args:
        folder (str | Path): Path to the dataset directory.

    Returns:
        str: The inferred dataset type as a string ("json", "pace", or "snap").

    Raises:
        ValueError: If no known file types are found in the folder.
    """
    folder = Path(folder)

    if any(folder.glob("*.json")):
        return "json"
    if any(folder.glob("*.gr")):
        return "pace"
    if any(folder.glob("*.txt")):
        return "snap"

    raise ValueError(
        f"Could not infer dataset type for folder {folder}. "
        f"No .json, .gr, or .txt files found."
    )


def load_instances(folder: str | Path) -> list[GraphInstance]:
    """
    Automatically infers the dataset format and loads all graph instances from a directory.

    Args:
        folder (str | Path): Path to the dataset directory.

    Returns:
        list[GraphInstance]: A list of loaded graph instances.
    """
    folder = Path(folder)
    dataset_type = infer_dataset_type(folder)

    if dataset_type == "json":
        return load_all_json_graphs(folder)
    elif dataset_type == "pace":
        return load_all_pace_graphs(folder)
    elif dataset_type == "snap":
        return load_all_snap_graphs(folder)
    
    return []
