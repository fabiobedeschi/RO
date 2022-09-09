from argparse import ArgumentParser

from src.solver import Solver
from src.utils import graph_from_json, print_graph_info, print_title


def _init_parser():
    arguments = [
        {
            "flags": ["-f", "--data-file"],
            "required": True,
            "type": str,
            "metavar": "FILE",
            "help": "Path to the data file.",
        },
        {
            "flags": ["-l", "--max-leaves"],
            "required": True,
            "type": int,
            "metavar": "N",
            "help": "Maximum number of leaves.",
        },
        {
            "flags": ["-r", "--root"],
            "type": str,
            "help": "Root node.",
        },
        {
            "flags": ["-i", "--max-iter"],
            "type": int,
            "help": "Maximum number of iterations.",
        },
        {
            "flags": ["-n", "--max-non-improving-iter"],
            "type": int,
            "help": "Maximum number of non improving iterations.",
        },
        {
            "flags": ["-p", "--leaf-penalty"],
            "type": float,
            "help": "Leaf penalty.",
        },
        {
            "flags": ["--no-hot-stop"],
            "action": "store_true",
            "help": "Disable hot stop.",
        },
        {
            "flags": ["-t", "--max-tabu-size"],
            "type": int,
            "help": "Maximum tabu size.",
        },
        {
            "flags": ["-T", "--initial-temp"],
            "type": float,
            "help": "Initial temperature.",
        },
        {
            "flags": ["-c", "--cooling-rate"],
            "type": str,
            "help": "Cooling rate.",
            "choices": ["linear", "exponential"],
        },
        {
            "flags": ["-C", "--cooling-factor"],
            "type": float,
            "help": "Cooling factor.",
        },
        {
            "flags": ["--population-size"],
            "type": int,
            "help": "Population size.",
        },
        {
            "flags": ["--no-breeding"],
            "action": "store_true",
            "help": "Disable breeding.",
        },
        {
            "flags": ["--no-mutation"],
            "action": "store_true",
            "help": "Disable mutation.",
        },
        {
            "flags": ["--no-elitism"],
            "action": "store_true",
            "help": "Disable elitism.",
        },
        {
            "flags": ["--breeding-rate"],
            "type": float,
            "help": "Breeding rate.",
        },
        {
            "flags": ["--mutation-rate"],
            "type": float,
            "help": "Mutation rate.",
        },
        {
            "flags": ["--elitism-rate"],
            "type": float,
            "help": "Elitism rate.",
        },
        {
            "flags": ["-d", "--debug"],
            "action": "store_true",
            "help": "Debug mode.",
        },
        {
            "flags": ["-m", "--multiprocess"],
            "action": "store_true",
            "help": "Multiprocess mode.",
        },
        {
            "flags": ["-u", "--cpu-count"],
            "type": int,
            "help": "Number of CPUs to use.",
        },
        {
            "flags": ["--mode"],
            "type": str,
            "help": "Search strategies to use.",
            "choices": ["greedy", "tabu", "sa", "genetic"],
            "nargs": "+",
        },
    ]

    _parser = ArgumentParser(
        description="Find the minimum leaf cost spanning tree of a graph."
    )
    for arg in arguments:
        flags = arg.pop("flags")
        _parser.add_argument(*flags, **arg)

    return _parser


def main(
    data_file,
    max_leaves,
    root=None,
    max_iter=None,
    max_non_improving_iter=None,
    leaf_penalty=None,
    hot_stop=None,
    max_tabu_size=None,
    initial_temperature=None,
    cooling_rate=None,
    cooling_factor=None,
    population_size=None,
    breeding=None,
    mutation=None,
    elitism=None,
    breeding_rate=None,
    mutation_rate=None,
    elitism_rate=None,
    debug=None,
    multiprocess=None,
    cpu_count=None,
    mode=None,
):
    g = graph_from_json(data_file)
    s = Solver(graph=g, multiprocess=multiprocess, cpu_count=cpu_count)

    print_title("MST")
    mst = s.find_mst()
    print_graph_info(mst)

    if mode is None or "greedy" in mode:
        print_title("MLCST (greedy)")
        greedy_mlcst = s.find_mlcst_greedy(
            max_leaves=max_leaves,
            root=root,
            max_iter=max_iter,
            max_non_improving_iter=max_non_improving_iter,
            leaf_penalty=leaf_penalty,
            hot_stop=hot_stop,
            debug=debug,
        )
        print_graph_info(greedy_mlcst)

    if mode is None or "tabu" in mode:
        print_title("MLCST (tabu)")
        tabu_mlcst = s.find_mlcst_tabu(
            max_leaves=max_leaves,
            root=root,
            max_iter=max_iter,
            max_non_improving_iter=max_non_improving_iter,
            max_tabu_size=max_tabu_size,
            leaf_penalty=leaf_penalty,
            hot_stop=hot_stop,
            debug=debug,
        )
        print_graph_info(tabu_mlcst)

    if mode is None or "sa" in mode:
        print_title("MLCST (simulated annealing)")
        tabu_mlcst = s.find_mlcst_sa(
            max_leaves=max_leaves,
            root=root,
            leaf_penalty=leaf_penalty,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            cooling_factor=cooling_factor,
            hot_stop=hot_stop,
            debug=debug,
        )
        print_graph_info(tabu_mlcst)

    if mode is None or "genetic" in mode:
        print_title("MLCST (genetic)")
        genetic_mlcst = s.find_mlcst_genetic(
            max_leaves=max_leaves,
            root=root,
            leaf_penalty=leaf_penalty,
            max_iter=max_iter,
            hot_stop=hot_stop,
            population_size=population_size,
            breeding=breeding,
            mutation=mutation,
            elitism=elitism,
            breeding_rate=breeding_rate,
            mutation_rate=mutation_rate,
            elitism_rate=elitism_rate,
            debug=debug,
        )
        print_graph_info(genetic_mlcst)


if "__main__" == __name__:
    parser = _init_parser()
    args = parser.parse_args()
    main(
        data_file=args.data_file,
        max_leaves=args.max_leaves,
        root=args.root,
        max_iter=args.max_iter,
        max_non_improving_iter=args.max_non_improving_iter,
        leaf_penalty=args.leaf_penalty,
        hot_stop=not args.no_hot_stop,
        max_tabu_size=args.max_tabu_size,
        initial_temperature=args.initial_temp,
        cooling_rate=args.cooling_rate,
        cooling_factor=args.cooling_factor,
        population_size=args.population_size,
        breeding=not args.no_breeding,
        mutation=not args.no_mutation,
        elitism=not args.no_elitism,
        breeding_rate=args.breeding_rate,
        mutation_rate=args.mutation_rate,
        elitism_rate=args.elitism_rate,
        debug=args.debug,
        multiprocess=args.multiprocess,
        cpu_count=args.cpu_count,
        mode=args.mode,
    )
