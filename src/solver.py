from src.graph import Graph


class Solver:
    """
    Abstract class for solvers.
    """

    def find_mst(self, graph: Graph) -> Graph:
        """
        Find the Minimum Spanning Tree on a given graph using Kruskal's algorithm.
        """
        weight_sorted_edges = sorted(graph.get_all_edges(), key=lambda e: e[2])
        mst = Graph(directed=graph.is_directed())
        for edge in weight_sorted_edges:
            if not mst.will_lead_to_cycle(edge[0], edge[1]):
                mst.add_edge(edge[0], edge[1], edge[2])
            if mst.is_spanning_tree(nodes_count=graph.get_node_count()):
                break
        return mst
