from trec_cds.data.load_data_from_file import load_jsonl
from trec_cds.features.build_features import ClinicalTrialsFeatures
import datetime
from trec_cds.features.index_clinical_trials import Indexer
from trec_cds.models.trec_evaluation import load_qrels, print_line, read_bm25
import pytrec_eval
from typing import List
from tqdm import tqdm
import logging
import pandas as pd
import json
import os

feature_builder = ClinicalTrialsFeatures(spacy_language_model_name='en_core_sci_lg')


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


def get_sections(dict_item, options):
    sections = []
    positive_keywords = ["_".join(x.split()) for x in dict_item['positive_entities']]
    negative_keywords = [f'no_{"_".join(x.split())}' for x in dict_item['negated_entities']]
    past_history_keywords = [f'PMH_{"_".join(x.split())}' for x in dict_item['pmh_entities']]
    family_keywords = [f'FH_{"_".join(x.split())}' for x in dict_item['fh_entities']]
    if 'positive' in options:
        sections.extend(positive_keywords)
    if 'negative' in options:
        sections.extend(negative_keywords)
    if 'pmh' in options:
        sections.extend(past_history_keywords)
    if 'fh' in options:
        sections.extend(family_keywords)
    return sections


def build_query(patient, options):
    sections = get_sections(patient, options=options)
    # input_text = f"{patient['current_medical_history']} {patient['conditions']} {patient['keywords']}"
    # input_text = f"{patient['description']} {patient['conditions']} {patient['cmh_keywords']}"
    input_text = f"{patient['description']} {patient['conditions']} {patient['keywords']}"
    text = feature_builder.preprocess_text(input_text, lemmatised=False)
    text.extend(sections)
    return text

def swap_exclusion(exclusion_dict):
    exclusion_dict['positive_entities_1'] = exclusion_dict["negated_entities"]
    exclusion_dict['negated_entities'] = exclusion_dict['positive_entities']
    exclusion_dict['positive_entities'] = exclusion_dict['positive_entities_1']
    exclusion_dict.pop('positive_entities_1', None)

    return exclusion_dict

def build_index_input(clinical_trial, options):
    exclusion_dict = swap_exclusion(exclusion_dict=clinical_trial['exclusion_criteria'])
    exclusion_sections = get_sections(exclusion_dict, options=options)

    sections = get_sections(clinical_trial['inclusion_criteria'], options=options)
    # input_text = f"{clinical_trial['brief_summary']} {clinical_trial['official_title']} {clinical_trial['brief_title']} {clinical_trial['detailed_description']} {' '.join(clinical_trial['conditions'])}  {' '.join(clinical_trial['inclusion'])}"
    input_text = f"{clinical_trial['brief_summary']} {clinical_trial['official_title']} {clinical_trial['brief_title']} {clinical_trial['detailed_description']} {' '.join(clinical_trial['conditions'])}  {clinical_trial['criteria']}"
    text = feature_builder.preprocess_text(input_text, lemmatised=False)
    text.extend(sections)
    text.extend(exclusion_sections)
    return text

if __name__ == '__main__':
    options = ['positive', 'negative', 'fh']  # 'pmh'
    print(options)
    # lemma = 'lemma'
    lemma = 'not_lemma'
    print(lemma)

    trials_file = "/newstorage4/wkusa/data/trec_cds/trials_parsed-new.jsonl"
    trials = load_jsonl(trials_file)
    print(len(trials))

    lookup_table = {x_index: x['nct_id'] for x_index, x in enumerate(trials)}

    index_sufix = f"{lemma}_pnf_eligibility_all-text_cmh-keywords"
    index_file = f"/newstorage4/wkusa/data/trec_cds/index_2022_all_{index_sufix}.p"
    indexer = Indexer()

    if os.path.isfile(index_file):
        print(f"loading index: {index_file}")
        indexer.load_index(filename=index_file)
        indexer.lookup_table = lookup_table
    else:
        print(f"creating index: {index_file}")
        clinical_trials_text: List[List[str]] = []
        for clinical_trial in tqdm(trials):
            text = build_index_input(clinical_trial=clinical_trial, options=options)
            text = [x.lower() for x in text if x.strip()]
            clinical_trials_text.append(text)

        indexer.index_text(text=clinical_trials_text, lookup_table=lookup_table)
        indexer.save_index(filename=index_file)

    for patient_file in ['topics2021', 'topics2022']:
        run_name = f"submission_{patient_file}_{index_sufix}"
        return_top_n = 3000
        submission_folder = '/newstorage4/wkusa/data/trec_cds/data/submissions/'

        infile = f"/home/wkusa/projects/TREC/trec-cds1/data/processed/{patient_file}.jsonl"
        additional_topics = f"/home/wkusa/projects/TREC/trec-cds1/data/interim/{patient_file}.csv"
        df = pd.read_csv(additional_topics)
        patients = load_jsonl(infile)

        for _patient_id in range(len(patients)):
            patients[_patient_id]['conditions'] = df.loc[_patient_id, "Conditions"]
            patients[_patient_id]['keywords'] = df.loc[_patient_id, "description_keywords"]
            patients[_patient_id]['cmh_keywords'] = df.loc[_patient_id, "keywords"]

        output_scores = {}
        for patient in tqdm(patients):
            doc = build_query(patient=patient, options=options)
            doc = [x.lower() for x in doc if x.strip()]

            output_scores[patient['patient_id']] = indexer.query_single(
                query=doc, return_top_n=return_top_n
            )

        with open(f"{submission_folder}/bm25p-{run_name}-{datetime.datetime.now().date()}.json", "w") as fp:
            json.dump(output_scores, fp)

        results_file = f"{submission_folder}/bm25p-{run_name}-{datetime.datetime.now().date()}"
        logging.info("Converting total number of %d topics", len(output_scores))
        with open(results_file, "w") as fp:
            for topic_no in output_scores:
                logging.info("working on topic: %s", topic_no)

                sorted_results = {
                    k: v
                    for k, v in sorted(
                        output_scores[topic_no].items(), key=lambda item: item[1], reverse=True
                    )
                }

                logging.info("normalizing results")
                max_value = max(sorted_results.values())
                sorted_results = {k: v / max_value for k, v in sorted_results.items()}

                for rank, doc in enumerate(sorted_results):
                    if rank >= 1000:  # TREC submission allows max top 1000 results
                        break
                    score = sorted_results[doc]

                    line = f"{topic_no} Q0 {doc} {rank + 1} {score} {run_name}\n"
                    fp.write(line)

        print('NDCG:')
        eval(run=read_bm25(results_file),
             qrels_path="/home/wkusa/projects/trec-cds/data/external/qrels2021.txt")
        print("\n\nprecison, RR:")
        eval(run=read_bm25(results_file),
             qrels_path="/home/wkusa/projects/TREC/trec-cds/data/external/qrels2021_binary.txt")
        print('\n\n')