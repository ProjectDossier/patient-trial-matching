import argparse
import json
import logging
import pickle
from typing import List, Dict

import numpy as np
import pytrec_eval
from rank_bm25 import BM25Okapi, BM25Plus
from tqdm import tqdm

from CTnlp.clinical_trial import ClinicalTrial
from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.patient import load_patients_from_xml
from CTnlp.patient import Patient
from trec_cds.features.build_features import ClinicalTrialsFeatures
from trec_cds.models.trec_evaluation import load_qrels, print_line, read_bm25
from trec_cds.features.index_clinical_trials import Indexer

def eval(run, qrels_path):
    qrels = load_qrels(qrels_path)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "P_10", "recip_rank", "ndcg_cut_5"}
    )
    results = evaluator.evaluate(run)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            pass
            # print_line(measure, query_id, value)

    for measure in sorted(query_measures.keys()):
        print_line(
            measure,
            "all",
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()],
            ),
        )


# class Indexer:
#     """Wrapper around BM25Okapi class that indexes ClinicalTrials and allows for
#     querying them with Topic data. input data must be preprocessed and tokenized."""
#
#     index: BM25Plus
#
#     def index_clinical_trials(self, text):
#         self.index = BM25Plus(text)
#
#     def query_single(self, query: List[str], return_top_n: int) -> Dict[str, float]:
#         topic_scores = {}
#         doc_scores = self.index.get_scores(query)
#         for index, score in zip(
#             np.argsort(doc_scores)[-return_top_n:], np.sort(doc_scores)[-return_top_n:]
#         ):
#             topic_scores[cts[index].nct_id] = score
#
#         return topic_scores


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
        default="/newstorage4/wkusa/data/trec_cds/data/processed/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--submission_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/submissions/",
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
        default=1000,
        type=int,
        help="return top n results from retrieval model",
    )

    args = parser.parse_args()

    cts = parse_clinical_trials_from_folder(
        folder_name=args.clinical_trials_folder, first_n=None
    )

    feature_builder = ClinicalTrialsFeatures(spacy_language_model_name='en_core_sci_lg')
    for clinical_trial in tqdm(cts):
        feature_builder.preprocess_clinical_trial(clinical_trial=clinical_trial)

    topics: List[Patient] = load_patients_from_xml(patient_file=args.topic_file)

    print("lowercase, no punctuation, no stopwords, no keywords")

    options = [
        "summary",
        "eligibility",
        "inclusion",
        "exclusion",
        "description",
        "description_criteria",
        "description_criteria_title",
        "summary_criteria",
        "summary_criteria_title",
        "summary_description_titles",
        "summary_description_titles_conditions",
        "summary_description_titles_conditions_inclusion",
        "summary_description_titles_conditions_eligibility",
        "brief_title",
        "official_title",
        "conditions"
        "all",
    ]

    for option in tqdm(options):
        print(option)
        cts_tokenized = []
        for _clinical_trial in cts:
            if option == "summary":
                cts_tokenized.append(feature_builder.preprocess_text(_clinical_trial.brief_summary))
            elif option == "brief_title":
                cts_tokenized.append(feature_builder.preprocess_text(_clinical_trial.brief_title))
            elif option == "official_title":
                cts_tokenized.append(feature_builder.preprocess_text(_clinical_trial.official_title))
            elif option == "eligibility":
                cts_tokenized.append(feature_builder.preprocess_text(_clinical_trial.criteria))
            elif option == "inclusion":
                cts_tokenized.append(feature_builder.preprocess_text(" ".join(_clinical_trial.inclusion)))
            elif option == "exclusion":
                cts_tokenized.append(feature_builder.preprocess_text(" ".join(_clinical_trial.exclusion)))
            elif option == "description":
                cts_tokenized.append(feature_builder.preprocess_text(_clinical_trial.detailed_description))
            elif option == "description_criteria":
                cts_tokenized.append(
                    feature_builder.preprocess_text(f"{_clinical_trial.detailed_description} {_clinical_trial.criteria}"))
            elif option == "description_criteria_title":
                cts_tokenized.append(
                    feature_builder.preprocess_text(f"{_clinical_trial.detailed_description} {_clinical_trial.criteria} {_clinical_trial.brief_title}"))
            elif option == "summary_criteria":
                cts_tokenized.append(feature_builder.preprocess_text(f"{_clinical_trial.brief_summary} {_clinical_trial.criteria}"))
            elif option == "summary_criteria_title":
                cts_tokenized.append(feature_builder.preprocess_text(f"{_clinical_trial.brief_summary} {_clinical_trial.criteria} {_clinical_trial.brief_title}"))
            elif option == "summary_description_titles":
                cts_tokenized.append(feature_builder.preprocess_text(
                    f"{_clinical_trial.brief_summary} {_clinical_trial.official_title} {_clinical_trial.brief_title} {_clinical_trial.detailed_description}"))
            elif option == "summary_description_titles_conditions":
                cts_tokenized.append(feature_builder.preprocess_text(
                    f"{_clinical_trial.brief_summary} {_clinical_trial.official_title} {_clinical_trial.brief_title} {_clinical_trial.detailed_description} {' '.join(_clinical_trial.conditions)}"))
            elif option == "summary_description_titles_conditions_inclusion":
                cts_tokenized.append(feature_builder.preprocess_text(
                    f"{_clinical_trial.brief_summary} {_clinical_trial.official_title} {_clinical_trial.brief_title} {_clinical_trial.detailed_description} {' '.join(_clinical_trial.conditions)} {' '.join(_clinical_trial.inclusion)}"))
            elif option == "summary_description_titles_conditions_eligibility":
                cts_tokenized.append(feature_builder.preprocess_text(
                    f"{_clinical_trial.brief_summary} {_clinical_trial.official_title} {_clinical_trial.brief_title} {_clinical_trial.detailed_description} {' '.join(_clinical_trial.conditions)}  {_clinical_trial.criteria}"))
            elif option == "conditions":
                cts_tokenized.append(feature_builder.preprocess_text(" ".join(_clinical_trial.conditions)))
            elif option == "all":
                cts_tokenized.append(_clinical_trial.text_preprocessed)
            else:
                continue

        cts_tokenized = [ct if len(ct) > 0 else ["empty"] for ct in cts_tokenized]

        for ct_index in range(len(cts_tokenized)):
            cts_tokenized[ct_index] = [x.lower() for x in cts_tokenized[ct_index] if x.strip()]

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

        with open(f"{args.results_folder}/bm25p-{option}-220824.json", "w") as fp:
            json.dump(output_scores, fp)

        results = output_scores

        logging.info("Converting total number of %d topics", len(output_scores))
        with open(f"{args.submission_folder}/bm25p-{option}-220824", "w") as fp:
            for topic_no in results:
                logging.info("working on topic: %s", topic_no)

                sorted_results = {
                    k: v
                    for k, v in sorted(
                        results[topic_no].items(), key=lambda item: item[1], reverse=True
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

        eval(run=read_bm25(f"{args.submission_folder}/bm25p-{option}-220824"),
             qrels_path="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt")

        eval(run=read_bm25(f"{args.submission_folder}/bm25p-{option}-220824"),
             qrels_path="/home/wkusa/projects/TREC/trec-cds/data/external/qrels2021_binary.txt")
