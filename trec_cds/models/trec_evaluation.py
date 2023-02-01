import pytrec_eval


def read_bm25(path):
    run = {}
    with open(path, "r") as run1:
        lines = run1.readlines()

        for line in lines:
            splitted = line.split(" ")

            query_id = splitted[0]
            doc_id = splitted[2]
            rank = int(splitted[3])
            score = float(splitted[4])

            if run.get(query_id):
                run.get(query_id).update({doc_id: score})
            else:
                run.update({query_id: {}})
                run.get(query_id).update({doc_id: score})
    return run


def load_qrels(qrels_path):
    # load the qrels
    qrels = {}
    with open(qrels_path, "r") as run1:
        lines = run1.readlines()

        for line in lines:
            splitted = line.split(" ")

            query_id = splitted[0]
            doc_id = splitted[2]
            score = int(splitted[3])

            if qrels.get(query_id):
                qrels.get(query_id).update({doc_id: score})
            else:
                qrels.update({query_id: {}})
                qrels.get(query_id).update({doc_id: score})
    return qrels


def print_line(measure, scope, value):
    print("{:25s}{:8s}{:.4f}".format(measure, scope, value))


def write_line(measure, scope, value):
    return "{:25s}{:8s}{:.4f}".format(measure, scope, value)





if __name__ == "__main__":
    qrels_path = "../../data/external/qrels2021.txt"

    run = read_bm25(
        "/home/wkusa/projects/trec-cds/data/processed/submissions/bm25-baseline-postprocessed-reranked-Mi"
        # "/Users/wojciechkusa/projects/shared-tasks/trec-cds/data/processed/submissions/simple_run"
    )

    qrels = load_qrels(qrels_path)

    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"}
    )
    results = evaluator.evaluate(run)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
                print_line(measure, query_id, value)

    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )
