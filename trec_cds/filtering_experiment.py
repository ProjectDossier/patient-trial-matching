import os

from trec_cds.data.load_data_from_file import load_jsonl
from trec_cds.data.trec_submission import convert_to_trac_submission
from trec_cds.models.postprocessing_2022 import create_new_filters, postprocessing
from trec_cds.models.trec_evaluation import load_qrels, print_line, read_bm25, write_line
from trec_cds.features.index_clinical_trials import Indexer
import pytrec_eval


def eval(run, qrels_path, eval_measures= {"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"}):
    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, eval_measures
    )
    results = evaluator.evaluate(run)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            pass
            # print_line(measure, query_id, value)
    output_string = ""
    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )
        output_string += write_line(measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),)

    return output_string

if __name__ == '__main__':

    lemma = 'not_lemma'

    submission_folder = '/newstorage4/wkusa/data/trec_cds/data/submissions/'

    trials_file = "/newstorage4/wkusa/data/trec_cds/trials_parsed-new.jsonl"
    trials = load_jsonl(trials_file)
    trials = create_new_filters(trials)

    cts_dict = {ct['nct_id']: ct for ct in trials}

    patient_file = 'topics2021'
    infile = f"/home/wkusa/projects/TREC/trec-cds1/data/processed/{patient_file}.jsonl"
    patients = load_jsonl(infile)

    # "bm25p-submission_topics2021_not_lemma_pnfp_eligibility_all-text_cmh-keywords-2022-08-28"
    output_file = ""
    processed_data_folder = "/newstorage4/wkusa/data/trec_cds/data/processed/ecir2023/ie/"
    run_files = os.listdir(processed_data_folder)
    for run_file in run_files:
        if not run_file.endswith('json'):
            continue
        print(run_file)

        # first_stage_results_file = f"{submission_folder}/bm25p-submission_{patient_file}_{lemma}-2022-08-28 12:05:34.101636.json"
        # first_stage_results_file = f"{submission_folder}/bm25p-submission_{patient_file}_{lemma}_pnf_eligibility_all-text_cmh-keywords-2022-08-28.json"
        # run_name = f"submission_{patient_file}_{lemma}"

        filtered_submission_json = f"{processed_data_folder}/filtered_{run_file}.json"
        run_name = f"filtered_{run_file[:-5]}"
        filtered_submission_trec = f"{processed_data_folder}/{run_name}"
        postprocessing(
            result_filename=f"{processed_data_folder}/{run_file}",
            output_file=filtered_submission_json,
            clinical_trials_dict=cts_dict,
            patients=patients,
        )

        convert_to_trac_submission(
            result_filename=filtered_submission_json,
            run_name=run_name,
            output_folder=processed_data_folder,
            trim_scores_less_than=0.10,
        )

        output_results = eval(run=read_bm25(filtered_submission_trec),
             qrels_path="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt",
                              eval_measures= {"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"})

        output_results += eval(run=read_bm25(filtered_submission_trec),
             qrels_path="/home/wkusa/projects/TREC/trec-cds/data/external/qrels2021_binary.txt",
                               eval_measures= {"P_10", "recip_rank"})

        with open(f"{filtered_submission_trec}-results", 'w') as fp:
            fp.write(output_results)
