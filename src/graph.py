from collections import defaultdict
from typing import Union, Optional


class Graph:
    """
    Create a graph from a list of weighted edges represented as tuples (node1, node2, weight).

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

    def __init__(self, edges=None, directed=False, coordinates=None):
        self._edges = defaultdict(set)
        self._directed = directed
        if edges is not None:
            self.add_edges(edges)
        if coordinates is not None:
            self._coordinates = coordinates

    def __eq__(self, other):
        """
        Check if two graphs are equal.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """
        Hash the graph.
        """
        return hash(tuple(self.get_all_edges()))

    def add_edges(self, edges):
        """
        Add multiple edges to the graph.
        """
        for node1, node2, weight in edges:
            self.add_edge(node1, node2, weight)
        return self

    def add_edge(self, node1, node2, weight):
        """
        Add a single edge to the graph.
        """
        self._edges[node1].add((node2, weight))
        if not self._directed:
            self._edges[node2].add((node1, weight))
        return self

    def remove_edges(self, edges):
        """
        Remove multiple edges from the graph.
        """
        for node1, node2, weight in edges:
            self.remove_edge(node1, node2, weight)
        return self

    def remove_edge(self, node1, node2, weight=None):
        """
        Remove an edge from the graph. If weight is provided, only the edge with that weight will be removed.
        """
        self._edges[node1].discard(
            (node2, weight or self.get_edge_weight(node1, node2))
        )
        if not self._directed:
            self._edges[node2].discard(
                (node1, weight or self.get_edge_weight(node2, node1))
            )
        return self

    def get_edges(self, node) -> set:
        """
        Get the edges starting from a given node.
        """
        return {(node, dest, weight) for dest, weight in self._edges[node]}

    def get_edge_weight(self, node1, node2) -> Optional[int]:
        """
        Get the weight of the edge between two given nodes, None if not found.
        """
        for _, node, weight in self.get_edges(node1):
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
        Get all the edges of the graph. Some edges may be omitted in undirected graphs.
        """
        edges = set()
        for edge in {
            (n1, n2, w)
            for n1 in self.get_all_nodes()
            for _, n2, w in self.get_edges(n1)
        }:
            if not self._directed:
                if edge[0] < edge[1]:
                    edges.add(edge)
            else:
                edges.add(edge)
        return edges

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
        Get the neighbors of the node. A neighbor of a node is another node that is directly connected to the first.
        """
        return {neighbor for _, neighbor, _ in self.get_edges(node)}

    def is_directed(self) -> bool:
        """
        Check if the graph is directed.
        """
        return self._directed

    def exists_path(self, node1, node2):
        """
        Check if a path exists between two nodes (BF). Can't guarantee it is the shortest path.
        """
        visited = set()
        stack = [node1]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            if node == node2:
                return True
            visited.add(node)
            stack.extend(self.get_neighbors(node))
        return False

    def find_path(self, node1, node2, visited=None) -> list:
        """
        Find a path between two nodes (DF). Can't guarantee it is the shortest path.
        """
        visited = visited or set()
        visited.add(node1)
        for neighbor in self.get_neighbors(node1):
            if neighbor == node2:
                return [node1, node2]
            if neighbor in visited:
                continue
            path = self.find_path(neighbor, node2, visited)
            if path:
                return [node1] + path

    def get_edges_in_path(self, path) -> set:
        """
        Get the edges in a path.
        """
        return {
            (path[i], path[i + 1], self.get_edge_weight(path[i], path[i + 1]))
            for i in range(len(path) - 1)
        }

    def is_cyclic(self) -> bool:
        """
        Check if the graph has at least a cycle in it.
        """
        for node in self.get_all_nodes():
            visited = set()
            stack = [node]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(self.get_neighbors(node))
                if node in stack:
                    return True
        return False

    def is_connected(self) -> bool:
        """
        Check if the graph is connected. Searches for a path between all pairs of different nodes.
        """
        return all(
            {
                self.exists_path(n1, n2)
                for n1 in self.get_all_nodes()
                for n2 in self.get_all_nodes()
                if n1 != n2
            }
        )

    def is_tree(self, nodes_count=None) -> bool:
        """
        Check if the graph is a tree. A tree is graph having N nodes, N-1 edges and no cycles.

        nodes_count:
            The number of nodes in the graph.
            Used to check for a tree while constructing from another graph.
            If not provided, the number of nodes will be inferred from the edges.
        """
        if nodes_count is None:
            nodes_count = self.get_node_count()
        return (self.get_edge_count() == nodes_count - 1) and not self.is_cyclic()

    def is_spanning_tree(self, nodes_count=None) -> bool:
        """
        Check if the graph is a spanning tree. A spanning tree is a tree that connects all the nodes.
        """
        return self.is_tree(nodes_count) and self.is_connected()

    def get_total_weight(self) -> Union[int, float]:
        """
        Get the total weight of the graph. Sum of all the edge weights.
        """
        return sum(weight for _, _, weight in self.get_all_edges())

    def get_leaf_nodes(self, nodes=None) -> set:
        """
        Get the all leaf nodes of the graph. A leaf node is a node with only one neighbor.
        """
        nodes = nodes or self.get_all_nodes()
        return {node for node in nodes if len(self.get_neighbors(node)) == 1}

    def get_leaf_nodes_from_root(self, root) -> set:
        """
        Get the all leaf nodes starting from a root. A leaf node is a node with only one neighbor.
        """
        return self.get_leaf_nodes(nodes=self.get_all_nodes() - {root})

    def get_leaf_node_count(self) -> int:
        """
        Get the number of leaf nodes in the graph.
        """
        return len(self.get_leaf_nodes())

    def get_leaf_node_count_from_root(self, root) -> int:
        """
        Get the number of leaf nodes starting from a root.
        """
        return len(self.get_leaf_nodes_from_root(root))

    def is_leaf_node(self, node) -> bool:
        """
        Check if the node is a leaf node.
        """
        return len(self.get_neighbors(node)) == 1

    def get_degree(self, node) -> int:
        """
        Get the degree of the node. The number of neighbors of the node.
        """
        return len(self.get_neighbors(node))

    def get_all_coordinates(self) -> dict:
        """
        Get the coordinates of the graph.
        """
        return self._coordinates

    def get_node_coordinates(self, node) -> Optional[tuple]:
        """
        Get the coordinates of the node. None if the node doesn't have coordinates.
        """
        if self._coordinates.get(node):
            return self._coordinates[node]["x"], self._coordinates[node]["y"]
        return None
