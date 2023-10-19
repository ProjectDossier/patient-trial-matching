import argparse
import copy
import datetime
import json
import logging
from typing import List, Dict, Any

from tqdm import tqdm

from trec_cds.data.load_data_from_file import load_jsonl
from trec_cds.lexical.features.build_features import ClinicalTrialsFeatures
from trec_cds.lexical.features.index_clinical_trials import Indexer
from trec_cds.trec_evaluation import read_bm25, evaluate

feature_builder = ClinicalTrialsFeatures(spacy_language_model_name="en_core_sci_lg")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NEGATIVE_ENTITIES = "negative"
AFFIRMATIVE_ENTITIES = "affirmative"


def get_sections(
    entities_dict: Dict[str, Dict[str, str]], options: List[str]
) -> List[str]:
    new_terms: List[str] = []
    for prefix, entity_key in {
        "cmh": "cmh_entities",
        "pmh": "pmh_entities",
        "fh": "fh_entities",
    }.items():
        for entity in entities_dict[entity_key]:
            if prefix not in options:
                continue

            if entity["negated"]:
                if NEGATIVE_ENTITIES not in options:
                    continue
                new_terms.append(
                    f'no_{prefix}_{"_".join(entity["text"].strip().split())}'
                )
            else:
                if AFFIRMATIVE_ENTITIES not in options:
                    continue
                new_terms.append(f'{prefix}_{"_".join(entity["text"].strip().split())}')

    return new_terms


def build_query(patient: Dict[str, Any], options: List[str]) -> List[str]:
    sections = get_sections(patient, options=options)
    text = feature_builder.preprocess_text(patient["description"], lemmatised=False)
    # text = feature_builder.preprocess_text(patient['current_medical_history'], lemmatised=True)
    text.extend(sections)
    return text


def swap_exclusion(
    exclusion_dict: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    out_dict = copy.deepcopy(exclusion_dict)
    for key in out_dict.keys():
        for entity_id in range(len(out_dict[key])):
            out_dict[key][entity_id]["negated"] = not out_dict[key][entity_id][
                "negated"
            ]
    return out_dict


def build_index_input(clinical_trial: Dict[str, Any], options: List[str]) -> List[str]:
    exclusion_dict = swap_exclusion(exclusion_dict=clinical_trial["exclusion_criteria"])
    exclusion_sections = get_sections(exclusion_dict, options=options)

    inclusion_sections = get_sections(
        clinical_trial["inclusion_criteria"], options=options
    )
    input_text = f"{clinical_trial['brief_summary']} {clinical_trial['official_title']} {clinical_trial['brief_title']} {clinical_trial['detailed_description']} {' '.join(clinical_trial['conditions'])}  {' '.join(clinical_trial['inclusion'])}"
    # input_text = f"{clinical_trial['brief_summary']} {clinical_trial['official_title']} {clinical_trial['brief_title']} {clinical_trial['detailed_description']} {' '.join(clinical_trial['conditions'])}  {clinical_trial['criteria']}"
    text = feature_builder.preprocess_text(input_text, lemmatised=False)
    text.extend(inclusion_sections)
    text.extend(exclusion_sections)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials_file",
        default="/newstorage4/wkusa/data/trec_cds/trials_parsed-jbi.jsonl",
        type=str,
        help="clinical trials parsed into a jsonl file",
    )
    parser.add_argument(
        "--topic_file",
        default="/home/wkusa/projects/trec-cds/data/external/topics2021.jsonl",
        type=str,
        help="path to a jsonl file with topics data",
    )
    parser.add_argument(
        "--results_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie/2021/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--submission_folder",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/jbi/ie/2021/",
        type=str,
        help="path to an outfile where indexed results will be saved.",
    )
    parser.add_argument(
        "--return_top_n",
        default=1000,
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

    runs_dict = {
        "an_cpf": [NEGATIVE_ENTITIES, AFFIRMATIVE_ENTITIES, "cmh", "pmh", "fh"],
        "an_cp": [NEGATIVE_ENTITIES, AFFIRMATIVE_ENTITIES, "cmh", "pmh"],
        "an_cf": [NEGATIVE_ENTITIES, AFFIRMATIVE_ENTITIES, "cmh", "fh"],
        "an_c": [NEGATIVE_ENTITIES, AFFIRMATIVE_ENTITIES, "cmh"],
        "a_cpf": [AFFIRMATIVE_ENTITIES, "cmh", "pmh", "fh"],
        "a_cp": [AFFIRMATIVE_ENTITIES, "cmh", "pmh"],
        "a_cf": [AFFIRMATIVE_ENTITIES, "cmh", "fh"],
        "a_c": [AFFIRMATIVE_ENTITIES, "cmh"],
        "n_cpf": [NEGATIVE_ENTITIES, "cmh", "pmh", "fh"],
        "n_cp": [NEGATIVE_ENTITIES, "cmh", "pmh"],
        "n_cf": [NEGATIVE_ENTITIES, "cmh", "fh"],
        "n_c": [NEGATIVE_ENTITIES, "cmh"],
    }

    logger.info("Loading trials from %s", args.trials_file)
    trials = load_jsonl(args.trials_file)
    logger.info("Loaded %d trials", len(trials))

    for run_name, options in runs_dict.items():
        logger.info("Running %s", run_name)
        logger.info("with options %s", options)

        logger.info("Loading patients from %s", args.topic_file)
        patients = load_jsonl(args.topic_file)
        print([patient["is_smoker"] for patient in patients])
        print([patient["is_drinker"] for patient in patients])

        logger.info("Expanding index for %s", run_name)
        clinical_trials_text: List[List[str]] = []
        for clinical_trial in tqdm(trials):
            text = build_index_input(clinical_trial=clinical_trial, options=options)
            text = [x.lower() for x in text if x.strip()]
            clinical_trials_text.append(text)
        logger.info("Index expansion complete")

        logger.info("Indexing %s", run_name)
        lookup_table = {x_index: x["nct_id"] for x_index, x in enumerate(trials)}
        indexer = Indexer()
        indexer.index_text(text=clinical_trials_text, lookup_table=lookup_table)
        logger.info("Indexing complete")

        logger.info("Querying %s", run_name)
        output_scores = {}
        for patient in patients:
            doc = build_query(patient=patient, options=options)
            doc = [x.lower() for x in doc if x.strip()]

            output_scores[patient["patient_id"]] = indexer.query_single(
                query=doc, return_top_n=args.return_top_n
            )
        logger.info("Querying complete")

        with open(f"{args.submission_folder}/bm25p-{run_name}-{TODAY}.json", "w") as fp:
            json.dump(output_scores, fp)

        run_output_file = f"{args.submission_folder}/bm25p-{run_name}-{TODAY}"
        run_results_file = f"{args.submission_folder}/bm25p-{run_name}-{TODAY}-results"

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

                    line = f"{topic_no} Q0 {doc} {rank + 1} {score} {run_name}\n"
                    fp.write(line)

        logger.info("Evaluating %s", run_name)
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

        logger.info("Evaluation results saved for %s", run_name)
