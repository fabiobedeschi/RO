from src.solver import Solver
from src.timer import timer
from src.utils import graph_from_json, print_graph_info, print_title

# TODO: Add command line arguments
DATA_FILE = "data/sparse_13.json"
MAX_LEAVES = 1
DEBUG = False


@timer
def main():
    g = graph_from_json(DATA_FILE)
    s = Solver(g)

    print_title("Graph info")
    print_graph_info(g)

    print_title("MST")
    mst = s.find_mst()
    print_graph_info(mst)

    print_title("MLCST (greedy)")
    greedy_mlcst = s.find_mlcst_greedy(
        max_leaves=MAX_LEAVES,
        root=None,
        max_iter=None,
        max_non_improving_iter=None,
        leaf_penalty=None,
        cost_function=None,
        debug=DEBUG,
    )
    print_graph_info(greedy_mlcst)

    print_title("MLCST (tabu)")
    tabu_mlcst = s.find_mlcst_tabu(
        max_leaves=MAX_LEAVES,
        root=None,
        max_iter=None,
        max_non_improving_iter=None,
        max_tabu_size=None,
        leaf_penalty=None,
        cost_function=None,
        debug=DEBUG,
    )
    print_graph_info(tabu_mlcst)

    print_title("MLCST (simulated annealing)")
    tabu_mlcst = s.find_mlcst_sa(
        max_leaves=MAX_LEAVES,
        root=None,
        leaf_penalty=None,
        cost_function=None,
        initial_temperature=None,
        cooling_rate=None,
        cooling_factor=None,
        hot_stop=False,
        debug=DEBUG,
    )
    print_graph_info(tabu_mlcst)


if "__main__" == __name__:
    main()
    # for nc in [5, 10, 20, 50, 100, 200, 500]:
    #     generate_random_data_file(nc, complete=True)
    #     generate_random_data_file(nc, complete=False)
