from src.graph import Graph


class Solver:
    """
    Solve different problem on a given graph
    """

    def __init__(self, graph: Graph):
        self._graph = graph

    def find_mst(self) -> Graph:
        """
        Find the Minimum Spanning Tree using Kruskal's algorithm.
        """
        mst = Graph(directed=self._graph.is_directed())
        for (n1, n2, w) in sorted(self._graph.get_all_edges(), key=lambda e: e[2]):
            if not mst.will_lead_to_cycle(n1, n2):
                mst.add_edge(n1, n2, w)
            if mst.is_spanning_tree(nodes_count=self._graph.get_node_count()):
                break
        return mst
