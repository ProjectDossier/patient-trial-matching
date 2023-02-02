import pytrec_eval


def read_bm25(path):
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


def load_qrels(qrels_path):
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


def print_line(measure, scope, value):
    print("{:25s}{:8s}{:.4f}".format(measure, scope, value))


def write_line(measure, scope, value):
    return f"{measure:25s}\t{scope:8s}\t{value:.4f}\n"


def eval(run, qrels_path, metrics={"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"}):
    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, metrics
    )
    results = evaluator.evaluate(run)
    scores = ""
    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            scores += write_line(measure, query_id, value)

    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )
    return scores


if __name__ == "__main__":
    new = "/newstorage4/wkusa/data/trec_cds/data/processed/ecir2023/ie/bm25p-keywords_experiment-anpf-i-2022-10-21 19:14:21.607104"
    baseline = "/newstorage4/wkusa/data/trec_cds/data/processed/ecir2023/bm25p-summary_description_titles_conditions_eligibility-221020"

    results_file = "/home/wkusa/projects/TREC/trec-cds1/data/processed/submissions/2022/BM25pe-f-pnf-akc-2021"
    # results_file = "/newstorage4/wkusa/data/trec_cds/data/submissions/bm25p-submission_topics2021_not_lemma_pnf_eligibility_all-text_cmh-keywords-2022-08-28"
    print('NDCG:')
    scores = eval(run=read_bm25(baseline),
         qrels_path="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt",
         metrics={"ndcg_cut_10", "ndcg_cut_5"})
    print("\n\nprecison, RR:")
    scores += eval(run=read_bm25(baseline),
         qrels_path="/home/wkusa/projects/TREC/trec-cds/data/external/qrels2021_binary.txt",
         metrics={"P_10", "recip_rank"})
    print('\n\n')


    with open("baseline.csv", 'w') as fp:
        fp.write(scores)
