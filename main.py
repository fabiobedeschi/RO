from src.solver import Solver
from src.timer import timer
from src.utils import graph_from_json


@timer
def main():
    print("=" * 80, "Progetto Ricerca Operativa", "=" * 80, sep="\n")

    g = graph_from_json("data/complete_02.json")

    print("MST:")
    s = Solver()
    mst = s.find_mst(g)

    print("edges:", mst.get_all_edges())
    print("edge count:", mst.get_edge_count())
    print("weight:", mst.get_total_weight())

    mst.remove_edge('01', '02')
    print("edges:", mst.get_all_edges())
    print("edge count:", mst.get_edge_count())


if "__main__" == __name__:
    main()
    # for nc in [5, 10, 20, 50, 100, 200, 500]:
    #     generate_random_data_file(nc, complete=True)
    #     generate_random_data_file(nc, complete=False)
