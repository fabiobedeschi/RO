from src.solver import Solver
from src.timer import timer
from src.utils import graph_from_json, print_graph_info, print_title


@timer
def main():
    g = graph_from_json("data/complete_04.json")
    s = Solver(g)
    print_title("Graph info")
    print_graph_info(g)

    mst = s.find_mst()
    print_title("MST")
    print_graph_info(mst)

    greedy_mlcst = s.find_mlcst(max_leaf_count=1, strategy="greedy")
    print_title("MLCST (greedy)")
    print_graph_info(greedy_mlcst)


if "__main__" == __name__:
    main()
    # for nc in [5, 10, 20, 50, 100, 200, 500]:
    #     generate_random_data_file(nc, complete=True)
    #     generate_random_data_file(nc, complete=False)
