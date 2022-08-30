from abc import ABC, abstractmethod

from src.graph import Graph


class Solver(ABC):
    """
    Abstract class for solvers.
    """

    @abstractmethod
    def solve(self, graph: Graph):
        """
        Solve on the given graph.
        """
        pass
