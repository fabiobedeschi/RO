from src.solver import Solver
from src.timer import timer
from src.utils import graph_from_json


@timer
def main():
    print("=" * 80, "Progetto Ricerca Operativa", "=" * 80, sep="\n")

    g = graph_from_json("data/graph_02.json")

    print("MST:")
    s = Solver(g)
    mst = s.find_mst()
    print("nodes:", mst.get_all_nodes())
    print("nodes count:", mst.get_node_count())
    print("edges:", mst.get_all_edges())
    print("edge count:", mst.get_edge_count())
    print("weight:", mst.get_total_weight())
    print("leaf nodes:", mst.get_leaf_nodes())
    print("leaf count:", mst.get_leaf_node_count())


if "__main__" == __name__:
    main()
    # for nc in [5, 10, 20, 50, 100, 200, 500]:
    #     generate_random_data_file(nc, complete=True)
    #     generate_random_data_file(nc, complete=False)
