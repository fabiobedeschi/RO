import json
from collections import defaultdict
from typing import Union


class Graph:
    """
    Create a graph from a list of weighted edges.

    Examples:
        # Undirected graph
        Graph(
            edges=[(0, 1, 2), (0, 2, 5), (1, 2, 4)],
        )

        # Directed graph
        Graph(
            edges=[(0, 2, 2), (0, 4, 5), (1, 2, 4), (1, 3, 1), (2, 3, 3), (3, 0, 2)],
            directed=True,
        )
    """

    @staticmethod
    def from_json(filename):
        """
        Create a graph from a json file.
        """
        with open(filename, mode="r") as f:
            data = json.load(f)

        edges = [(n1, n2, w) for n1 in data["edges"].keys() for n2, w in data["edges"][n1].items()]
        directed = data["directed"]
        return Graph(edges, directed)

    def __init__(self, edges, directed=False):
        self._edges = defaultdict(set)
        self._directed = directed
        self.add_edges(edges)

    def add_edges(self, connections):
        """
        Add multiple edges to the graph.
        """
        for node1, node2, weight in connections:
            self.add_edge(node1, node2, weight)

    def add_edge(self, node1, node2, weight):
        """
        Add a single edge to the graph.
        """
        self._edges[node1].add((node2, weight))
        if not self._directed:
            self._edges[node2].add((node1, weight))

    def get_edges(self, node) -> set:
        """
        Get the edges of the node.
        """
        return self._edges[node]

    def get_edge_weight(self, node1, node2) -> Union[int, None]:
        """
        Get the weight of the edge, None if not found.
        """
        for node, weight in self.get_edges(node1):
            if node == node2:
                return weight
        return None

    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        """
        count = sum(len(self._edges[node]) for node in self._edges)
        return count if self._directed else count // 2

    def get_all_edges(self) -> set:
        """
        Get all the edges of the graph. Some edges may be duplicated in undirected graphs.
        """
        return {(node1, node2, weight) for node1 in self.get_all_nodes() for node2, weight in self.get_edges(node1)}

    def get_all_nodes(self) -> set:
        """
        Get the nodes of the graph.
        """
        return set(self._edges.keys())

    def get_node_count(self) -> int:
        """
        Get the number of nodes in the graph.
        """
        return len(self._edges)

    def get_neighbors(self, node) -> set:
        """
        Get the neighbors of the node.
        """
        return {node for node, weight in self.get_edges(node)}

    def is_directed(self) -> bool:
        """
        Check if the graph is directed.
        """
        return self._directed
