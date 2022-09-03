import json
import os
from itertools import combinations
from math import sqrt
from random import random, randint

from fastjsonschema import validate

from src.graph import Graph
from src.schema import schema


def graph_from_json(filename):
    """
    Create a graph from a json data file.
    """
    with open(filename, mode="r") as f:
        data = json.load(f)

    validate(data=data, definition=schema)

    edges = [
        (n1, n2, w)
        for n1 in data["edges"].keys()
        for n2, w in data["edges"][n1].items()
    ]
    directed = data["directed"]
    return Graph(edges, directed)


def print_title(title: str, line_len: int = 80, centered: bool = True):
    """
    Print a title in console.
    """
    print(
        "=" * line_len,
        f"{' ' * (centered * (line_len - len(title)) // 2)}{title}",
        "=" * line_len,
        sep="\n",
    )


def print_graph_info(g: Graph, root: str = None):
    """
    Print information about a graph in console.
    """
    root = root or sorted(g.get_all_nodes())[0]
    print()
    print("nodes count:", g.get_node_count())
    print("weight:", g.get_total_weight())
    print("is tree:", g.is_tree())
    print("root:", root)
    print("leaf node from root:", g.get_leaf_nodes_from_root(root))
    print("leaf count from root:", g.get_leaf_node_count_from_root(root))
    print("degree of root node:", g.get_degree(root))
    print("edge count:", g.get_edge_count())
    print("edges:", g.get_all_edges())


def generate_random_data_file(node_count, complete=True, max_dim=100, filename=None):
    """
    Generate a random undirected graph data file.
    """
    z_fill_count = len(str(node_count))
    nodes = [
        (str(i).zfill(z_fill_count), randint(0, max_dim), randint(0, max_dim))
        for i in range(node_count)
    ]
    edges = []
    for ((n1, x1, y1), (n2, x2, y2)) in combinations(nodes, 2):
        if complete or random() > 0.5:
            if n1 < n2:
                edge = (n1, n2, round(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 3))
                edges.append(edge)

    data = {"directed": False, "edges": {}, "plot": {}}
    for (n, *_) in nodes:
        data["edges"][n] = {n2: w for (n1, n2, w) in edges if n2 > n == n1}
    for (n, x, y) in nodes:
        data["plot"][n] = {"x": x, "y": y}

    if filename is None:
        f = []
        prefix = ""
        for (_, _, filenames) in os.walk("data/"):
            for filename in filenames:
                prefix = "complete" if complete else "sparse"
                if filename.startswith(f"{prefix}_"):
                    f.append(filename)
            break
        filename = f"data/{prefix}_{str(len(f) + 1).zfill(2)}.json"

    with open(filename, mode="w") as f:
        json.dump(data, f)
