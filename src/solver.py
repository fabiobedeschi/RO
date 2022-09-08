import os
import random
from copy import deepcopy
from math import exp
from typing import Callable, Optional, Union

from multiprocess import Pool
from tqdm import tqdm

from src.graph import Graph
from src.timer import timer
from src.utils import loop_generator

Number = Union[int, float]


class Solver:
    """
    Solve different problems on a given graph
    """

    def __init__(self, graph: Graph, multiprocess: bool = False, cpu_count: int = None):
        self._graph = graph
        self._multiprocess = multiprocess
        if self._multiprocess:
            self._pool = Pool(processes=cpu_count or os.cpu_count())

    def _explore_neighbourhood(self, t: Graph, root: str, cost_function: Callable):
        swap_operations = set()

        # Loop over the edges in the graph who are not in the current solution
        for (n1, n2, w) in self._graph.get_all_edges():
            if (n1, n2, w) in t.get_all_edges():
                continue

            if not any(leaf in {n1, n2} for leaf in t.get_leaf_nodes_from_root(root)):
                # If none of the nodes in the edge is a leaf node, skip it
                continue

            cycle_edges = t.get_edges_in_path(path=t.find_path(n1, n2))
            for (cn1, cn2, cw) in cycle_edges:
                # Calculate the cost of the neighbor solution after the swap
                tmp_mlcst = deepcopy(t)
                tmp_mlcst.add_edge(n1, n2, w)
                tmp_mlcst.remove_edge(cn1, cn2, cw)
                swap_operations.add(
                    ((n1, n2, w), (cn1, cn2, cw), cost_function(tmp_mlcst))
                )

        return swap_operations

    def _explore_neighbourhood_multiprocess(
        self, t: Graph, root: str, cost_function: Callable
    ):
        def _calculate_swap_gain(_e_in, _e_out, _t, _cf):
            _tmp_t = deepcopy(_t)
            _tmp_t.add_edge(*_e_in)
            _tmp_t.remove_edge(*_e_out)
            return _e_in, _e_out, _cf(_tmp_t)

        def _filter_swap_edges(_e_in, _t, _root):
            _candidate_swaps = set()

            if _e_in in _t.get_all_edges():
                return _candidate_swaps
            if not any(
                leaf in {_e_in[0], _e_in[1]}
                for leaf in _t.get_leaf_nodes_from_root(_root)
            ):
                return _candidate_swaps

            for _e_out in _t.get_all_edges():
                _candidate_swaps.add((_e_in, _e_out))

            return _candidate_swaps

        swap_operations = set()
        candidate_swaps = set()

        for candidate_swap in self._pool.starmap(
            _filter_swap_edges,
            [(e_in, t, root) for e_in in self._graph.get_all_edges()],
        ):
            candidate_swaps.update(candidate_swap)

        for swap_operation in self._pool.starmap(
            _calculate_swap_gain,
            [(e_in, e_out, t, cost_function) for e_in, e_out in candidate_swaps],
        ):
            swap_operations.add(swap_operation)

        return swap_operations

    def find_mst(self, verbose=True) -> Graph:
        """
        Find the Minimum Spanning Tree using Kruskal's algorithm.
        """
        mst = Graph(directed=self._graph.is_directed())
        for (n1, n2, w) in tqdm(
            sorted(self._graph.get_all_edges(), key=lambda e: e[2]),
            desc="Finding MST",
            disable=not verbose
        ):
            if not mst.exists_path(n1, n2):
                mst.add_edge(n1, n2, w)
            if mst.is_spanning_tree(nodes_count=self._graph.get_node_count()):
                break
        return mst

    @timer
    def find_mlcst_greedy(
        self,
        max_leaves: int,
        root: Optional[str] = None,
        max_iter: Optional[int] = None,
        max_non_improving_iter: Optional[int] = None,
        leaf_penalty: Optional[Number] = None,
        cost_function: Optional[Callable] = None,
        hot_stop: bool = False,
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
        :param hot_stop: If True, the algorithm will immediately stops if the current respect the condition on the leaves, defaults to False
        :param debug: Whether to print debug information, defaults to False

        :return: The leaf constrained spanning tree
        """

        # Initialize search hyper-parameters
        root = root or sorted(self._graph.get_all_nodes())[0]
        max_iter = max_iter or self._graph.get_node_count() * 100
        max_non_improving_iter = max_non_improving_iter or max_iter // 10
        leaf_penalty = leaf_penalty or max(
            map(lambda e: e[2], self._graph.get_all_edges())
        )
        cost_function = cost_function or (
            lambda t: t.get_total_weight()
            + leaf_penalty
            * (
                t.get_leaf_node_count_from_root(root) - max_leaves
                if t.get_leaf_node_count_from_root(root) > max_leaves
                else 0
            )
        )

        # Initialize the current solution with the MST of the graph
        mlcst = self.find_mst(verbose=False)

        # Initialize working variables
        iter_count = 0
        non_improving_iter_count = 0
        current_optimum = cost_function(mlcst)

        for _ in tqdm(
            loop_generator(),
            total=max_iter,
            desc="Finding MLCST",
            disable=debug,
        ):
            # Break out condition
            if (
                iter_count >= max_iter
                or non_improving_iter_count >= max_non_improving_iter
            ):
                break

            if debug:
                print(
                    f"Iteration {iter_count + 1} / {max_iter} "
                    f"(non-improving: {non_improving_iter_count} / {max_non_improving_iter}) "
                    f"(optimum: {current_optimum}) "
                )

            iter_count += 1
            non_improving_iter_count += 1

            if hot_stop and mlcst.get_leaf_node_count_from_root(root) <= max_leaves:
                # If the number of leaf nodes is less than or equal to the max leaf count, exit the loop
                if debug:
                    print(
                        f"Found solution with {mlcst.get_leaf_node_count_from_root(root)} leaf nodes"
                    )
                break

            # Find the best operation to perform exploring the neighborhood of the current solution
            # Save elements as (edge_in, edge_out, cost_after_swap)
            if self._multiprocess:
                swap_operations = self._explore_neighbourhood_multiprocess(
                    t=mlcst, root=root, cost_function=cost_function
                )
            else:
                swap_operations = self._explore_neighbourhood(
                    t=mlcst, root=root, cost_function=cost_function
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

    @timer
    def find_mlcst_tabu(
        self,
        max_leaves: Optional[int],
        root: Optional[str] = None,
        max_iter: Optional[int] = None,
        max_non_improving_iter: Optional[int] = None,
        leaf_penalty: Optional[Number] = None,
        cost_function: Optional[Callable] = None,
        max_tabu_size: Optional[int] = None,
        hot_stop: bool = False,
        debug: bool = False,
    ) -> Graph:
        """
        Find the Minimum Leaf Constrained Spanning Tree using a tabu search strategy.

        :param max_leaves: The maximum number of leaf nodes allowed in the tree
        :param root: The root node of the tree, defaults to the node with the lowest identifier will be used
        :param max_iter: The maximum number of iterations in total, defaults to number of nodes in the graph times 100
        :param max_non_improving_iter: The maximum number of non-improving iterations in a row, defaults to max_iter / 10
        :param max_tabu_size: The maximum size of the tabu list (FIFO stack), defaults to half of the number of edges in the graph
        :param leaf_penalty: The penalty for each leaf node in the tree (lower means more leaf nodes can be tolerated), defaults to the number of nodes in the graph
        :param cost_function: The cost function to use to evaluate the solution, defaults to the selected edges weight plus the number of exceeding leaf nodes times the leaf penalty
        :param hot_stop: If True, the algorithm will immediately stops if the current respect the condition on the leaves, defaults to False
        :param debug: Whether to print debug information, defaults to False

        :return: The leaf constrained spanning tree
        """

        # Initialize search hyper-parameters
        root = root or sorted(self._graph.get_all_nodes())[0]
        max_iter = max_iter or self._graph.get_node_count() * 100
        max_non_improving_iter = max_non_improving_iter or max_iter // 10
        max_tabu_size = max_tabu_size or self._graph.get_edge_count() // 2
        leaf_penalty = leaf_penalty or max(
            map(lambda e: e[2], self._graph.get_all_edges())
        )
        cost_function = cost_function or (
            lambda t: t.get_total_weight()
            + leaf_penalty
            * (
                t.get_leaf_node_count_from_root(root) - max_leaves
                if t.get_leaf_node_count_from_root(root) > max_leaves
                else 0
            )
        )

        # Initialize the current solution with the MST of the graph
        mlcst = self.find_mst(verbose=False)
        best_mlcst = deepcopy(mlcst)

        # Initialize working variables
        iter_count = 0
        non_improving_iter_count = 0
        tabu_list = []
        current_optimum = cost_function(mlcst)
        found_new_optimum = False

        for _ in tqdm(
            loop_generator(),
            total=max_iter,
            desc=f"Finding MLCST",
            disable=debug,
        ):
            # Break out condition
            if (
                iter_count >= max_iter
                or non_improving_iter_count >= max_non_improving_iter
            ):
                break

            if debug:
                print(
                    f"Iteration {iter_count + 1} / {max_iter} "
                    f"(non-improving: {non_improving_iter_count} / {max_non_improving_iter}) "
                    f"(optimum: {current_optimum}) "
                )
            iter_count += 1
            non_improving_iter_count += 1

            if hot_stop and mlcst.get_leaf_node_count_from_root(root) <= max_leaves:
                # If the number of leaf nodes is less than or equal to the max leaf count, exit the loop
                if debug:
                    print(
                        f"Found solution with {mlcst.get_leaf_node_count_from_root(root)} leaf nodes"
                    )
                best_mlcst = deepcopy(mlcst)
                break

            # Find the best operation to perform exploring the neighborhood of the current solution
            # Save elements as (edge_in, edge_out, cost_after_swap)
            if self._multiprocess:
                swap_operations = self._explore_neighbourhood_multiprocess(
                    t=mlcst, root=root, cost_function=cost_function
                )
            else:
                swap_operations = self._explore_neighbourhood(
                    t=mlcst, root=root, cost_function=cost_function
                )

            # Find the best swap operation by sorting the swap operations by the calculated objective function
            best_swap = None
            for candidate_best_swap in sorted(swap_operations, key=lambda e: e[2]):
                if candidate_best_swap[2] < current_optimum:
                    # Aspiration criteria
                    # If the swap operation can lead to a new optimum, use it
                    best_swap = candidate_best_swap
                    found_new_optimum = True
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
            if found_new_optimum:
                best_mlcst = deepcopy(mlcst)
                found_new_optimum = False

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

        return best_mlcst

    @timer
    def find_mlcst_sa(
        self,
        max_leaves: int,
        root: Optional[int] = None,
        leaf_penalty: Optional[Number] = None,
        cost_function: Optional[Callable] = None,
        initial_temperature: Optional[Number] = None,
        cooling_rate: Optional[str] = None,
        cooling_factor: Optional[Number] = None,
        hot_stop: bool = False,
        debug: bool = False,
    ) -> Graph:
        """
        Find the leaf constrained spanning tree of the graph using simulated annealing

        :param max_leaves: The maximum number of leaf nodes allowed in the tree
        :param root: The root node of the tree, defaults to the node with the lowest identifier will be used
        :param leaf_penalty: The penalty for each leaf node in the tree (lower means more leaf nodes can be tolerated), defaults to the highest weight of any edge in the graph
        :param cost_function: The cost function to use to evaluate the solution, defaults to the selected edges weight plus the number of exceeding leaf nodes times the leaf penalty
        :param initial_temperature: The initial temperature of the simulated annealing algorithm, defaults to the number of nodes in the graph times 100
        :param cooling_rate: The cooling rate of the simulated annealing algorithm, defaults to "exponential"
        :param cooling_factor: The cooling factor of the simulated annealing algorithm, default value depend on the cooling rate chosen
        :param hot_stop: If True, the algorithm will immediately stops if the current respect the condition on the leaves, defaults to False
        :param debug: Whether to print debug information, defaults to False
        """

        # Initialize search hyper-parameters
        root = root or sorted(self._graph.get_all_nodes())[0]
        leaf_penalty = leaf_penalty or max(
            map(lambda e: e[2], self._graph.get_all_edges())
        )
        cost_function = cost_function or (
            lambda t: t.get_total_weight()
            + leaf_penalty
            * (
                t.get_leaf_node_count_from_root(root) - max_leaves
                if t.get_leaf_node_count_from_root(root) > max_leaves
                else 0
            )
        )
        temperature = initial_temperature or self._graph.get_node_count() * 100
        if temperature <= 1:
            raise ValueError("Initial temperature must be greater than 1")

        cooling_rate = (cooling_rate or "exponential").lower()
        _known_cooling_rates = {"exponential", "linear"}
        if cooling_rate not in _known_cooling_rates:
            raise ValueError(
                f"Cooling rate must be one of {', '.join(_known_cooling_rates)}"
            )

        if cooling_rate == "exponential":
            cooling_factor = cooling_factor or 0.9
            if cooling_factor <= 0 or cooling_factor >= 1:
                raise ValueError("Cooling rate must be in between 0 and 1 excluded")
        if cooling_rate == "linear":
            cooling_factor = cooling_factor or 1
            if cooling_factor <= 0:
                raise ValueError("Cooling rate must be greater than 0")

        # Initialize the current solution with the MST of the graph
        mlcst = self.find_mst(verbose=False)
        best_mlcst = deepcopy(mlcst)

        # Initialize working variables
        iter_count = 0
        non_improving_iter_count = 0
        current_optimum = cost_function(mlcst)

        for _ in tqdm(loop_generator(), desc="Finding MLCST", disable=debug):
            # Break out condition
            if temperature < 1:
                break

            if debug:
                print(
                    f"Iteration {iter_count + 1} "
                    f"(T: {temperature}) "
                    f"(optimum: {current_optimum}) "
                )
            iter_count += 1
            non_improving_iter_count += 1

            if hot_stop and mlcst.get_leaf_node_count_from_root(root) <= max_leaves:
                # If the number of leaf nodes is less than or equal to the max leaf count, exit the loop
                # This speeds up the search sacrificing the quality of the solution
                if debug:
                    print(
                        f"Found solution with {mlcst.get_leaf_node_count_from_root(root)} leaf nodes"
                    )
                best_mlcst = deepcopy(mlcst)
                break

            # Explore the neighborhood of the current solution
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

                    if cost_function(tmp_mlcst) < current_optimum:
                        # If the neighbor solution is better than the current solution, use it
                        mlcst = tmp_mlcst
                        best_mlcst = deepcopy(tmp_mlcst)
                        current_optimum = cost_function(tmp_mlcst)
                        if debug:
                            print(f"\tUsing swap: {(n1, n2, w)} -> {(cn1, cn2, cw)}")
                            print(f"\tNew optimum found: {current_optimum}")
                        break

                    # If the neighbor solution is not better than the current solution,
                    # use it with a probability that decreases with the temperature
                    if random.random() < exp(
                        (current_optimum - cost_function(tmp_mlcst)) / temperature
                    ):
                        if debug:
                            print(f"\tUsing swap: {(n1, n2, w)} -> {(cn1, cn2, cw)}")
                        mlcst = tmp_mlcst
                        break

                else:
                    # Only continue if the inner loop was not broken...
                    continue
                # ...otherwise, break the outer loop
                break

            # Decrease the temperature
            if cooling_rate == "exponential":
                temperature *= cooling_factor
            elif cooling_rate == "linear":
                temperature -= cooling_factor

        return best_mlcst
