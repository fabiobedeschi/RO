from copy import deepcopy
from typing import Union, Callable

from src.graph import Graph


class Solver:
    """
    Solve different problems on a given graph
    """

    def __init__(self, graph: Graph):
        self._graph = graph
        self._stack = None

    def find_mst(self) -> Graph:
        """
        Find the Minimum Spanning Tree using Kruskal's algorithm.
        """
        mst = Graph(directed=self._graph.is_directed())
        for (n1, n2, w) in sorted(self._graph.get_all_edges(), key=lambda e: e[2]):
            if not mst.exists_path(n1, n2):
                mst.add_edge(n1, n2, w)
            if mst.is_spanning_tree(nodes_count=self._graph.get_node_count()):
                break
        return mst

    def find_mlcst_greedy(
        self,
        max_leaves: int,
        root: str = None,
        max_iter: int = None,
        max_non_improving_iter: int = None,
        leaf_penalty: Union[int, float] = None,
        cost_function: Callable = None,
        debug: bool = False,
    ) -> Graph:
        """
        Find the Minimum Leaf Constrained Spanning Tree using a greedy search strategy. Can stuck in local minima.

        :param max_leaves: The maximum number of leaf nodes allowed in the tree
        :param root: The root node of the tree, defaults to the node with the lowest identifier will be used
        :param max_iter: The maximum number of iterations in total, defaults to number of nodes in the graph times 100
        :param max_non_improving_iter: The maximum number of non-improving iterations in a row, defaults to max_iter / 10
        :param leaf_penalty: The penalty for each leaf node in the tree (lower means more leaf nodes can be tolerated), defaults to the number of nodes in the graph
        :param cost_function: The cost function to use to evaluate the solution, defaults to the selected edges weight plus the number of exceeding leaf nodes times the leaf penalty
        :param debug: Whether to print debug information, defaults to False

        :return: The leaf constrained spanning tree
        """

        # Initialize search hyper-parameters
        root = root or sorted(self._graph.get_all_nodes())[0]
        max_iter = max_iter or self._graph.get_node_count() * 100
        max_non_improving_iter = max_non_improving_iter or max_iter // 10
        leaf_penalty = leaf_penalty or self._graph.get_node_count()
        cost_function = cost_function or (
            lambda t: t.get_total_weight()
            + leaf_penalty * (t.get_leaf_node_count_from_root(root) - max_leaves)
        )

        # Initialize the current solution with the MST of the graph
        mlcst = self.find_mst()

        # Initialize working variables
        iter_count = 0
        non_improving_iter_count = 0
        current_optimum = cost_function(mlcst)

        while (
            iter_count < max_iter and non_improving_iter_count < max_non_improving_iter
        ):
            if debug:
                print(
                    f"Iteration {iter_count + 1} / {max_iter} "
                    f"(non-improving: {non_improving_iter_count + 1} / {max_non_improving_iter})"
                )
            iter_count += 1
            non_improving_iter_count += 1

            if mlcst.get_leaf_node_count_from_root(root) <= max_leaves:
                # If the number of leaf nodes is less than or equal to the max leaf count, exit the loop
                if debug:
                    print(
                        f"Found solution with {mlcst.get_leaf_node_count_from_root(root)} leaf nodes"
                    )
                break

            # Find the best operation to perform exploring the neighborhood of the current solution
            # Save elements as (edge_in, edge_out, cost_after_swap)
            swap_operations = set()

            # Loop over the edges in the graph who are not in the current solution
            for (n1, n2, w) in self._graph.get_all_edges():
                if (n1, n2, w) in mlcst.get_all_edges():
                    continue

                if not any(
                    leaf in {n1, n2} for leaf in mlcst.get_leaf_nodes_from_root(root)
                ):
                    # If none of the nodes in the edge is a leaf node, skip it
                    continue

                cycle_edges = mlcst.get_edges_in_path(path=mlcst.find_path(n1, n2))
                for (cn1, cn2, cw) in cycle_edges:
                    # Calculate the cost of the neighbor solution after the swap
                    tmp_mlcst = deepcopy(mlcst)
                    tmp_mlcst.add_edge(n1, n2, w)
                    tmp_mlcst.remove_edge(cn1, cn2, cw)
                    swap_operations.add(
                        ((n1, n2, w), (cn1, cn2, cw), cost_function(tmp_mlcst))
                    )

            # Find the best swap operation by sorting the swap operations by the calculated objective function
            best_swap = None
            for candidate_best_swap in sorted(swap_operations, key=lambda e: e[2]):
                if candidate_best_swap[2] < current_optimum:
                    # If the swap operation can lead to a new optimum, use it
                    best_swap = candidate_best_swap
                    current_optimum = candidate_best_swap[2]
                    non_improving_iter_count = 0
                    if debug:
                        print(f"\tNew optimum found: {current_optimum}")
                    break

                best_swap = candidate_best_swap
                break

            if best_swap is None:
                # If no swap operation can be performed, exit the loop
                if debug:
                    print("\tNo swap operation can be performed")
                break

            # Apply the selected swap operation
            if debug:
                print(
                    f"\tPerforming swap operation: {best_swap[0]} -> {best_swap[1]} (cost: {best_swap[2]})"
                )
            mlcst.add_edge(*best_swap[0])
            mlcst.remove_edge(*best_swap[1])

        if debug:
            if iter_count == max_iter:
                print("Reached maximum number of iterations")
            elif non_improving_iter_count == max_non_improving_iter:
                print("Reached maximum number of non-improving iterations")

        return mlcst

    def find_mlcst_tabu(
        self,
        max_leaves: int,
        root: str = None,
        max_iter: int = None,
        max_non_improving_iter: int = None,
        max_tabu_size: int = None,
        leaf_penalty: Union[int, float] = None,
        cost_function: Callable = None,
        debug: bool = False,
    ) -> Graph:
        """
        Find the Minimum Leaf Constrained Spanning Tree using a tabu search strategy.

        :param max_leaves: The maximum number of leaf nodes allowed in the tree
        :param root: The root node of the tree, defaults to the node with the lowest identifier will be used
        :param max_iter: The maximum number of iterations in total, defaults to number of nodes in the graph times 100
        :param max_non_improving_iter: The maximum number of non-improving iterations in a row, defaults to max_iter / 10
        :param max_tabu_size: The maximum size of the tabu list (FIFO stack), defaults to half of the number of nodes in the graph
        :param leaf_penalty: The penalty for each leaf node in the tree (lower means more leaf nodes can be tolerated), defaults to the number of nodes in the graph
        :param cost_function: The cost function to use to evaluate the solution, defaults to the selected edges weight plus the number of exceeding leaf nodes times the leaf penalty
        :param debug: Whether to print debug information, defaults to False

        :return: The leaf constrained spanning tree
        """

        # Initialize search hyper-parameters
        root = root or sorted(self._graph.get_all_nodes())[0]
        max_iter = max_iter or self._graph.get_node_count() * 100
        max_non_improving_iter = max_non_improving_iter or max_iter // 10
        max_tabu_size = max_tabu_size or self._graph.get_node_count() // 2
        leaf_penalty = leaf_penalty or self._graph.get_node_count()
        cost_function = cost_function or (
            lambda t: t.get_total_weight()
            + leaf_penalty * (t.get_leaf_node_count_from_root(root) - max_leaves)
        )

        # Initialize the current solution with the MST of the graph
        mlcst = self.find_mst()

        # Initialize working variables
        iter_count = 0
        non_improving_iter_count = 0
        tabu_list = []
        current_optimum = cost_function(mlcst)

        while (
            iter_count < max_iter and non_improving_iter_count < max_non_improving_iter
        ):
            if debug:
                print(
                    f"Iteration {iter_count + 1} / {max_iter} "
                    f"(non-improving: {non_improving_iter_count} / {max_non_improving_iter}) "
                    f"(current optimum: {current_optimum})"
                )
            iter_count += 1
            non_improving_iter_count += 1

            if mlcst.get_leaf_node_count_from_root(root) <= max_leaves:
                # If the number of leaf nodes is less than or equal to the max leaf count, exit the loop
                if debug:
                    print(
                        f"Found solution with {mlcst.get_leaf_node_count_from_root(root)} leaf nodes"
                    )
                break

            # Find the best operation to perform exploring the neighborhood of the current solution
            # Save elements as (edge_in, edge_out, cost_after_swap)
            swap_operations = set()

            # Loop over the edges in the graph who are not in the current solution
            for (n1, n2, w) in self._graph.get_all_edges():
                if (n1, n2, w) in mlcst.get_all_edges():
                    continue

                if not any(
                    leaf in {n1, n2} for leaf in mlcst.get_leaf_nodes_from_root(root)
                ):
                    # If none of the nodes in the edge is a leaf node, skip it
                    continue

                cycle_edges = mlcst.get_edges_in_path(path=mlcst.find_path(n1, n2))
                for (cn1, cn2, cw) in cycle_edges:
                    # Calculate the cost of the neighbor solution after the swap
                    tmp_mlcst = deepcopy(mlcst)
                    tmp_mlcst.add_edge(n1, n2, w)
                    tmp_mlcst.remove_edge(cn1, cn2, cw)
                    swap_operations.add(
                        ((n1, n2, w), (cn1, cn2, cw), cost_function(tmp_mlcst))
                    )

            # Find the best swap operation by sorting the swap operations by the calculated objective function
            best_swap = None
            for candidate_best_swap in sorted(swap_operations, key=lambda e: e[2]):
                if candidate_best_swap[2] < current_optimum:
                    # Aspiration criteria
                    # If the swap operation can lead to a new optimum, use it
                    best_swap = candidate_best_swap
                    current_optimum = candidate_best_swap[2]
                    non_improving_iter_count = 0
                    if debug:
                        print(f"\tNew optimum found: {current_optimum}")
                    break

                out_n1, out_n2, out_w = candidate_best_swap[1]
                if any(
                    tabu_el in {(out_n1, out_n2, out_w), (out_n2, out_n1, out_w)}
                    for tabu_el in tabu_list
                ):
                    # If the leaving edge is in the tabu list skip it
                    if debug:
                        print(f"\tSkipping tabu swap: {candidate_best_swap}")
                    continue

                best_swap = candidate_best_swap
                break

            if best_swap is None:
                # If no swap operation can be performed, exit the loop
                if debug:
                    print("\tNo swap operation can be performed")
                break

            # Apply the selected swap operation
            if debug:
                print(
                    f"\tPerforming swap operation: {best_swap[0]} -> {best_swap[1]} (cost: {best_swap[2]})"
                )
            mlcst.add_edge(*best_swap[0])
            mlcst.remove_edge(*best_swap[1])

            # Add entering edge to the tabu list
            tabu_list.append(best_swap[0])

            # Remove the oldest element from the tabu list if it is too long
            if len(tabu_list) > max_tabu_size:
                tabu_list.pop(0)

        if debug:
            if iter_count == max_iter:
                print("Reached maximum number of iterations")
            elif non_improving_iter_count == max_non_improving_iter:
                print("Reached maximum number of non-improving iterations")

        return mlcst
