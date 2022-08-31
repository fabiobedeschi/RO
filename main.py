from src.graph import Graph
from src.solver import Solver


def main():
    print("=" * 80, "Progetto Ricerca Operativa", "=" * 80, sep="\n")

    g = Graph.from_json("data/graph_01.json")

    print("MST:")
    s = Solver()
    mst = s.find_mst(g)

    print(mst.get_all_edges())
    print(mst.get_edge_count())


if "__main__" == __name__:
    main()
