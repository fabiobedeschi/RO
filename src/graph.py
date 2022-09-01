from collections import defaultdict
from typing import Union


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

    def __init__(self, edges=None, directed=False):
        self._edges = defaultdict(set)
        self._directed = directed
        if edges is not None:
            self.add_edges(edges)

    def add_edges(self, edges):
        """
        Add multiple edges to the graph.
        """
        for node1, node2, weight in edges:
            self.add_edge(node1, node2, weight)

    def add_edge(self, node1, node2, weight):
        """
        Add a single edge to the graph.
        """
        self._edges[node1].add((node2, weight))
        if not self._directed:
            self._edges[node2].add((node1, weight))

    def remove_edges(self, edges):
        """
        Remove multiple edges from the graph.
        """
        for node1, node2, weight in edges:
            self.remove_edge(node1, node2, weight)

    def remove_edge(self, node1, node2, weight=None):
        """
        Remove an edge from the graph.
        """
        self._edges[node1].discard((node2, weight or self.get_edge_weight(node1, node2)))
        if not self._directed:
            self._edges[node2].discard((node1, weight or self.get_edge_weight(node2, node1)))

    def get_edges(self, node) -> set:
        """
        Get the edges starting from a given node.
        """
        return {(node, dest, weight) for dest, weight in self._edges[node]}

    def get_edge_weight(self, node1, node2) -> Union[int, None]:
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
        for edge in {(n1, n2, w) for n1 in self.get_all_nodes() for _, n2, w in self.get_edges(n1)}:
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
        Get the neighbors of the node.
        """
        return {neighbor for _, neighbor, _ in self.get_edges(node)}

    def is_directed(self) -> bool:
        """
        Check if the graph is directed.
        """
        return self._directed

    def exists_path(self, node1, node2):
        """
        Check if a path exists between two nodes. Can't guarantee it is the shortest path.
        """
        visited = set()
        stack = [node1]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.get_neighbors(node))
            if node == node2:
                return True
        return False

    def has_cycle(self) -> bool:
        """
        Check if the graph has a cycle.
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

    def will_lead_to_cycle(self, node1, node2) -> bool:
        """
        Check if adding an edge between two nodes will lead to a cycle.
        """
        edges = self.get_all_edges()
        edges.add((node1, node2, 1))
        g = Graph(edges=edges, directed=self._directed)
        return g.has_cycle()

    def is_connected(self) -> bool:
        """
        Check if the graph is connected.
        """
        return all({self.exists_path(n1, n2) for n1 in self.get_all_nodes() for n2 in self.get_all_nodes() if n1 != n2})

    def is_tree(self, nodes_count=None) -> bool:
        """
        Check if the graph is a tree.

        nodes_count:
            The number of nodes in the graph.
            Used to check for a tree while constructing from another graph.
            If not provided, the number of nodes will be inferred from the edges.
        """
        if nodes_count is None:
            nodes_count = self.get_node_count()
        return (self.get_edge_count() == nodes_count - 1) and not self.has_cycle()

    def is_spanning_tree(self, nodes_count=None) -> bool:
        """
        Check if the graph is a spanning tree.

        nodes_count:
            The number of nodes in the graph.
            Used to check for a spanning tree while constructing from another graph.
            If not provided, the number of nodes will be inferred from the edges.
        """
        return self.is_tree(nodes_count) and self.is_connected()

    def get_total_weight(self) -> Union[int, float]:
        """
        Get the total weight of the graph.
        """
        return sum(weight for _, _, weight in self.get_all_edges())
