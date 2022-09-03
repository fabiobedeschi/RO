from src.solver import Solver
from src.timer import timer
from src.utils import graph_from_json, print_graph_info, print_title


@timer
def main():
    g = graph_from_json("data/graph_02.json")
    s = Solver(g)
    print_title("Graph info")
    print_graph_info(g)

    print_title("MST")
    mst = s.find_mst()
    print_graph_info(mst)

    print_title("MLCST (greedy)")
    greedy_mlcst = s.find_mlcst(max_leaf_count=1, strategy="greedy", max_iter=1000)
    print_graph_info(greedy_mlcst)

    print_title("MLCST (tabu)")
    tabu_mlcst = s.find_mlcst_tabu(
        max_leaf_count=1,
        root=None,
        max_iter=None,
        max_non_improving_iter=None,
        max_tabu_size=None,
        leaf_penalty=None,
        cost_function=None,
        debug=True,
    )
    print_graph_info(tabu_mlcst)


if "__main__" == __name__:
    main()
    # for nc in [5, 10, 20, 50, 100, 200, 500]:
    #     generate_random_data_file(nc, complete=True)
    #     generate_random_data_file(nc, complete=False)
