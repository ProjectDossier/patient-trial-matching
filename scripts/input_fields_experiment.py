import argparse
import datetime
import json
import logging
from typing import List

from tqdm import tqdm

from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.patient import Patient
from CTnlp.patient import load_patients_from_xml
from trec_cds.lexical.features.build_features import ClinicalTrialsFeatures
from trec_cds.lexical.features.index_clinical_trials import Indexer
from trec_cds.trec_evaluation import read_bm25, evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical_trials_folder",
        default="/newstorage4/wkusa/data/trec_cds/ClinicalTrials",
        type=str,
        help="path to a folder with clinical trials",
    )
    parser.add_argument(
        "--topic_file",
        default="/newstorage4/wkusa/data/trec_cds/data/external/topics2021.xml",
        type=str,
        help="path to an xml file with topics data",
    )
    parser.add_argument(
        "--results_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/sections/2021/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--submission_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/sections/2021/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--first_n",
        default=25000,
        type=int,
        help="load only first n clinical trial documents (max is ~370k)",
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

    TODAY: str = datetime.datetime.now().strftime("%Y%m%d")

    logger.info("Loading clinical trials")
    cts = parse_clinical_trials_from_folder(
        folder_name=args.clinical_trials_folder, first_n=args.first_n
    )

    logger.info("Preprocessing clinical trials")
    feature_builder = ClinicalTrialsFeatures(spacy_language_model_name="en_core_sci_lg")

    logger.info("Loading patients from %s", args.topic_file)
    topics: List[Patient] = load_patients_from_xml(patient_file=args.topic_file)

    options = {
        "brief_title": [x.brief_title for x in cts],
        "official_title": [x.official_title for x in cts],
        "conditions": [" ".join(x.conditions) for x in cts],
        "titles": [f"{x.brief_title} {x.official_title}" for x in cts],
        "inclusion": [" ".join(x.inclusion) for x in cts],
        "exclusion": [" ".join(x.exclusion) for x in cts],
        "eligibility": [x.criteria for x in cts],
        "summary": [x.brief_summary for x in cts],
        "description": [x.detailed_description for x in cts],
        "description_criteria": [f"{x.detailed_description} {x.criteria}" for x in cts],
        "description_criteria_title": [
            f"{x.detailed_description} {x.criteria} {x.brief_title}" for x in cts
        ],
        "description_criteria_titles": [
            f"{x.detailed_description} {x.criteria} {x.brief_title} {x.official_title}"
            for x in cts
        ],
        "summary_criteria": [f"{x.brief_summary} {x.criteria}" for x in cts],
        "summary_criteria_title": [
            f"{x.brief_summary} {x.criteria} {x.brief_title}" for x in cts
        ],
        "summary_criteria_titles": [
            f"{x.brief_summary} {x.criteria} {x.brief_title} {x.official_title}"
            for x in cts
        ],
        "summary_description_titles": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description}"
            for x in cts
        ],
        "summary_description_titles_conditions": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description} {' '.join(x.conditions)}"
            for x in cts
        ],
        "summary_description_titles_conditions_inclusion": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description} {' '.join(x.conditions)} {' '.join(x.inclusion)}"
            for x in cts
        ],
        "summary_description_titles_conditions_exclusion": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description} {' '.join(x.conditions)} {' '.join(x.exclusion)}"
            for x in cts
        ],
        "summary_description_titles_conditions_eligibility": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description} {' '.join(x.conditions)}  {x.criteria}"
            for x in cts
        ],
        "summary_description_titles_eligibility": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description}  {x.criteria}"
            for x in cts
        ],
        "summary_description_titles_inclusion": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description}  {' '.join(x.inclusion)}"
            for x in cts
        ],
        "summary_description_titles_exclusion": [
            f"{x.brief_summary} {x.official_title} {x.brief_title} {x.detailed_description}  {' '.join(x.exclusion)}"
            for x in cts
        ],
    }

    for option, cts_list in options.items():
        logger.info("Indexing %s", option)
        cts_tokenized = []

        for _clinical_trial in tqdm(cts_list):
            cts_tokenized.append(
                feature_builder.preprocess_text(
                    _clinical_trial,
                    no_stopwords=True,
                    no_punctuation=True,
                    lemmatised=False,
                )
            )

        cts_tokenized = [ct if len(ct) > 0 else ["empty"] for ct in cts_tokenized]

        for ct_index in range(len(cts_tokenized)):
            cts_tokenized[ct_index] = [
                x.lower() for x in cts_tokenized[ct_index] if x.strip()
            ]

        lookup_table = {x_index: x.nct_id for x_index, x in enumerate(cts)}
        indexer = Indexer()
        indexer.index_text(text=cts_tokenized, lookup_table=lookup_table)
        logger.info("Indexing complete")

        logger.info("Querying %s", option)
        output_scores = {}
        for topic in topics:
            doc = feature_builder.preprocess_text(
                topic.description,
                no_stopwords=True,
                no_punctuation=True,
                lemmatised=False,
            )
            doc = [x.lower() for x in doc if x.strip()]
            output_scores[topic.patient_id] = indexer.query_single(
                query=doc, return_top_n=args.return_top_n
            )
        logger.info("Querying complete")

        with open(f"{args.results_folder}/bm25p-{option}-{TODAY}.json", "w") as fp:
            json.dump(output_scores, fp)

        run_output_file = f"{args.submission_folder}/bm25p-{option}-{TODAY}"
        run_results_file = f"{args.submission_folder}/bm25p-{option}-{TODAY}-results"

        logging.info("Converting total number of %d topics", len(output_scores))
        with open(run_output_file, "w") as fp:
            for topic_no in output_scores:

                sorted_results = {
                    k: v
                    for k, v in sorted(
                        output_scores[topic_no].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                max_value = max(sorted_results.values())
                sorted_results = {k: v / max_value for k, v in sorted_results.items()}

                for rank, doc in enumerate(sorted_results):
                    if rank >= 1000:  # TREC submission allows max top 1000 results
                        break
                    score = sorted_results[doc]

                    line = f"{topic_no} Q0 {doc} {rank + 1} {score} {option}\n"
                    fp.write(line)

        logger.info("Evaluating %s", option)
        output_results = evaluate(
            run=read_bm25(run_output_file),
            qrels_path=args.graded_qrels,
            eval_measures={"ndcg_cut_5", "ndcg_cut_10"},
        )

        output_results += evaluate(
            run=read_bm25(run_output_file),
            qrels_path=args.binary_qrels,
            eval_measures={"P_10", "recip_rank"},
        )

        with open(run_results_file, "w") as fp:
            fp.write(output_results)

        logger.info("Evaluation results saved for %s", option)
