import argparse
from typing import Dict, Optional

import pytrec_eval


def read_bm25(path: str) -> Dict:
    """Reads BM25 run."""
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


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """load the qrels"""
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


def print_line(measure: str, scope: str, value: float) -> None:
    print("{:25s}{:8s}{:.4f}".format(measure, scope, value))


def write_line(measure: str, scope: str, value: float) -> str:
    return f"{measure:25s}\t{scope:8s}\t{value:.4f}\n"


def evaluate(run, qrels_path, eval_measures: Optional[Dict] = None) -> str:
    """Evaluates a run against the qrels.
    It prints the aggregated results and returns string with all.
    todo: problem when we want to evaluate multi graded relevance and binary then results are not ordered by query

    :param run:
    :param qrels_path:
    :param eval_measures:
    :return: string containing per query scores and aggregated scores
    """
    if eval_measures is None:
        eval_measures = {"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"}

    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, eval_measures)
    results = evaluator.evaluate(run)
    query_scores = ""
    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            query_scores += write_line(measure, query_id, value)

    aggregated_scores = ""
    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )
        aggregated_scores += write_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )

    return query_scores + aggregated_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, default="run file")
    parser.add_argument(
        "--binary_qrels",
        type=str,
        default="/home/wkusa/projects/trec-cds/data/external/qrels2021_binary.txt",
        help="path to the binary qrels file",
    )
    parser.add_argument(
        "--graded_qrels",
        type=str,
        default="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt",
        help="path to the graded qrels file",
    )

    parser.add_argument(
        "--results_file", type=str, required=True, help="path to the results csv file"
    )

    args = parser.parse_args()

    print("NDCG:")
    scores = evaluate(run=read_bm25(args.run), qrels_path=args.graded_qrels,
                      eval_measures={"ndcg_cut_10", "ndcg_cut_5"})
    print("\n\nprecison, RR:")
    scores += evaluate(run=read_bm25(args.run), qrels_path=args.binary_qrels, eval_measures={"P_10", "recip_rank"})
    print("\n\n")

    with open(args.results_file, "w") as fp:
        fp.write(scores)
