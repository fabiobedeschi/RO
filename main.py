from src.graph import Graph


def main():
    print("=" * 80, "Progetto Ricerca Operativa", "=" * 80, sep="\n")

    g = Graph.from_json("data/graph_01.json")
    print(g.get_edges("a"))


if "__main__" == __name__:
    main()
