import argparse
import logging
import os

from trec_cds.data.load_data_from_file import load_jsonl
from trec_cds.data.trec_submission import convert_to_trec_submission
from trec_cds.models.postprocessing import create_new_filters, postprocessing
from trec_cds.models.trec_evaluation import read_bm25, evaluate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials_file",
        default="/newstorage4/wkusa/data/trec_cds/trials_parsed-new.jsonl",
        type=str,
        help="clinical trials parsed into a jsonl file",
    )
    parser.add_argument(
        "--topic_file",
        default="/home/wkusa/projects/trec-cds/data/processed/topics2021.jsonl",
        type=str,
        help="path to a jsonl file with topics data",
    )
    parser.add_argument(
        "--runs_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie/2021/",
        type=str,
        help="path to folder which contains runs for filtering experiment.",
    )
    parser.add_argument(
        "--output_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie_filtered/2021/",
        type=str,
        help="path to an outfile where indexed results and evaluations will be saved.",
    )
    parser.add_argument(
        "--return_top_n",
        default=500,
        type=int,
        help="return top n results from retrieval model",
    )
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
    args = parser.parse_args()

    filtering_options = {
        "age": ["age"],
        "gender": ["gender"],
        "age_gender": ["age", "gender"],
        "smoking": ["smoking"],
        "drinking": ["drinking"],
        "smoking_drinking": ["smoking", "drinking"],
        "age_gender_smoking_drinking": ["age", "gender", "smoking", "drinking"],
    }

    logger.info("Loading clinical trials and patients data")
    trials = load_jsonl(args.trials_file)
    trials = create_new_filters(trials)
    cts_dict = {ct["nct_id"]: ct for ct in trials}

    patients = load_jsonl(args.topic_file)
    logger.info("Loaded clinical trials and patients data")

    run_files = os.listdir(args.runs_folder)
    for run_file in run_files:
        # we only work with raw json files, other files are trec submissions or evaluation results
        if not run_file.endswith("json"):
            continue
        run = f"{args.runs_folder}/{run_file}"
        run_name = f"filtered_{run_file[:-5]}"

        logger.info("Evaluating unfiltered run: %s", run_name)
        evaluate(
            run=read_bm25(f"{args.runs_folder}/{run_name}"),
            qrels_path=args.graded_qrels,
            eval_measures={"ndcg_cut_5", "ndcg_cut_10"},
        )
        evaluate(
            run=read_bm25(f"{args.runs_folder}/{run_name}"),
            qrels_path=args.binary_qrels,
            eval_measures={"P_10", "recip_rank"},
        )
        logger.info("Finished evaluating unfiltered run: %s", run_name)

        logger.info(f"Filtering run: {run_name}")

        for option_name, options in filtering_options.items():

            filtered_submission_json = f"{args.output_folder}/{run_name}_{option_name}.json"
            filtered_submission_trec = f"{args.output_folder}/{run_name}_{option_name}"
            filtered_results_file = f"{args.output_folder}/{run_name}_{option_name}-results"
            postprocessing(
                result_filename=run,
                output_file=filtered_submission_json,
                clinical_trials_dict=cts_dict,
                patients=patients,
                options=options
            )
            convert_to_trec_submission(
                result_filename=filtered_submission_json,
                run_name=run_name,
                output_folder=args.output_folder,
                trim_scores_less_than=0.10,
            )
            logger.info("Finished filtering run: %s\t%s", (run_name, option_name))

            logger.info("Evaluating filtered run: %s\t%s", (run_name, option_name))
            output_results = evaluate(
                run=read_bm25(filtered_submission_trec),
                qrels_path=args.graded_qrels,
                eval_measures={"ndcg_cut_5", "ndcg_cut_10"},
            )
            output_results += evaluate(
                run=read_bm25(filtered_submission_trec),
                qrels_path=args.binary_qrels,
                eval_measures={"P_10", "recip_rank"},
            )

            with open(filtered_results_file, "w") as fp:
                fp.write(output_results)
            logger.info("Finished evaluating filtered run: %s\t%s", (run_name, option_name))
