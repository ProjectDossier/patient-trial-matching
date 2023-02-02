import argparse
import json
import logging
from typing import List

from tqdm import tqdm

from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.patient import Patient
from CTnlp.patient import load_patients_from_xml
from trec_cds.features.build_features import ClinicalTrialsFeatures
from trec_cds.features.index_clinical_trials import Indexer
from trec_cds.models.trec_evaluation import read_bm25, evaluate

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
        default="/newstorage4/wkusa/data/trec_cds/data/processed/ecir2023/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--submission_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/ecir2023/",
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

    args = parser.parse_args()

    cts = parse_clinical_trials_from_folder(
        folder_name=args.clinical_trials_folder, first_n=None
    )

    feature_builder = ClinicalTrialsFeatures(spacy_language_model_name="en_core_sci_lg")
    for clinical_trial in tqdm(cts):
        feature_builder.preprocess_clinical_trial(clinical_trial=clinical_trial)

    topics: List[Patient] = load_patients_from_xml(patient_file=args.topic_file)

    print("lowercase, no punctuation, no stopwords, no keywords")

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
        "all": [x.text_preprocessed for x in cts],
    }

    for option, cts_list in options.items():
        print(f"\n{option=}")
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

        output_scores = {}
        for topic in topics:
            doc = feature_builder.preprocess_text(topic.description)
            doc = [x.lower() for x in doc if x.strip()]
            output_scores[topic.patient_id] = indexer.query_single(
                query=doc, return_top_n=args.return_top_n
            )

        with open(f"{args.results_folder}/bm25p-{option}-221020.json", "w") as fp:
            json.dump(output_scores, fp)

        results = output_scores

        logging.info("Converting total number of %d topics", len(output_scores))
        with open(f"{args.submission_folder}/bm25p-{option}-221020", "w") as fp:
            for topic_no in results:
                logging.info("working on topic: %s", topic_no)

                sorted_results = {
                    k: v
                    for k, v in sorted(
                        results[topic_no].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }

                logging.info("normalizing results")
                max_value = max(sorted_results.values())
                sorted_results = {k: v / max_value for k, v in sorted_results.items()}

                for rank, doc in enumerate(sorted_results):
                    if rank >= 1000:  # TREC submission allows max top 1000 results
                        break
                    score = sorted_results[doc]

                    line = f"{topic_no} Q0 {doc} {rank + 1} {score} {option}\n"
                    fp.write(line)

        output_results = evaluate(
            run=read_bm25(f"{args.submission_folder}/bm25p-{option}-221020"),
            qrels_path="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt",
            eval_measures={"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"},
        )

        output_results += evaluate(
            run=read_bm25(f"{args.submission_folder}/bm25p-{option}-221020"),
            qrels_path="/home/wkusa/projects/TREC/trec-cds/data/external/qrels2021_binary.txt",
            eval_measures={"P_10", "recip_rank"},
        )

        with open(f"{args.submission_folder}/bm25p-{option}-221020-results", "w") as fp:
            fp.write(output_results)
