from src.graph import Graph


class Solver:
    """
    Solve different problems on a given graph
    """

    def __init__(self, graph: Graph, greedy_max_iterations: int = 1000):
        self._graph = graph
        self._greedy_max_iterations = greedy_max_iterations

    def _sort_edges_by_weight(self, reverse=False) -> list:
        return sorted(self._graph.get_all_edges(), key=lambda e: e[2], reverse=reverse)

    def find_mst(self) -> Graph:
        """
        Find the Minimum Spanning Tree using Kruskal's algorithm.
        """
        mst = Graph(directed=self._graph.is_directed())
        for (n1, n2, w) in self._sort_edges_by_weight():
            if not mst.exists_path(n1, n2):
                mst.add_edge(n1, n2, w)
            if mst.is_spanning_tree(nodes_count=self._graph.get_node_count()):
                break
        return mst

    def find_mlcst(self, max_leaf_count: int, root=None, strategy=None) -> Graph:
        """
        Find the Minimum Leaf Constrained Spanning Tree using the given strategy.
        """
        if max_leaf_count < 1:
            raise ValueError("The max leaf count must be at least 1")

        root = root or sorted(self._graph.get_all_nodes())[0]
        match strategy:
            case "greedy":
                return self._find_mlcst_greedy(max_leaf_count, root)
            case _:
                return self._find_mlcst_greedy(max_leaf_count, root)

    def _find_mlcst_greedy(self, max_leaf_count, root) -> Graph:
        """
        Find the Minimum Leaf Constrained Spanning Tree using a greedy strategy. Can get stuck in local minima.

        Start from a valid MST and try to reduce the number of leaf nodes by connecting them to their neighbors.
        This will decrease the number of leaf nodes but induce a cycle, which can be eliminated by removing the edge
        with the highest weight in it.
        In the end this is a 2-switch local search.
        """
        mlcst = self.find_mst()
        if mlcst.get_leaf_node_count_from_root(root) <= max_leaf_count:
            # If the number of leaf nodes is already less than or equal to the max leaf count, return the MST
            return mlcst

        # Initialize a stack with edges sorted by weight in descending order
        stack = self._sort_edges_by_weight(reverse=True)
        iter_count = 0
        while stack:
            iter_count += 1
            if iter_count > self._greedy_max_iterations:
                print(f"WARNING: Reached max iterations! ({self._greedy_max_iterations})")
                print(f"WARNING: Current solution could be a local minimum")
                print()
                break
            edge = (n1, n2, w) = stack.pop()
            if edge in mlcst.get_all_edges():
                # If the edge is already in the tree, skip it
                continue
            if n1 in mlcst.get_leaf_nodes_from_root(root):
                # If one of the nodes in the edge is a leaf node, add the edge to the tree

                # First remove the edge with the highest weight in the future cycle
                heaviest_edge = max(
                    self._graph.get_edges_in_path(path=mlcst.find_path(n1, n2)),
                    key=lambda e: e[2],
                )
                mlcst.remove_edge(*heaviest_edge)

                # Then add the new edge to the tree
                mlcst.add_edge(*edge)

                # Insert the removed edge back into the stack
                stack.append(heaviest_edge)

                if mlcst.get_leaf_node_count_from_root(root) <= max_leaf_count:
                    break

        return mlcst
