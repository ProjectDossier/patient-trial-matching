import argparse
import logging
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import (
    load_topics_from_xml,
    parse_clinical_trials_from_folder,
)
from trec_cds.data.topic import Topic
from trec_cds.features.ner import EntityRecognition, get_ner_model, get_displacy_options
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import json
import logging

import pandas as pd

from trec_cds.data.parsers import (
    load_topics_from_xml,
    parse_clinical_trials_from_folder,
)
from trec_cds.data.utils import Gender
from trec_cds.features.ner import EntityRecognition
import numpy as np
import scipy


def reranking(result_filename: str, output_file, clinical_trials_dict, topics):
    with open(result_filename) as fp:
        results = json.load(fp)

    model = SentenceTransformer("sentence-transformers/allenai-specter")

    total_checked = 0
    total_excluded = 0

    output_scores = {}
    for topic_no in results:
        included = {}
        logging.info(topic_no)
        excluded_num = 0
        checked = 0

        topic_encoded = model.encode([topics[int(topic_no) - 1].text])

        inclusions_similarity = {}
        exclusions_similarity = {}
        combined_score = {}
        for nct_id, score in results[topic_no].items():
            clinical_trial = clinical_trials_dict.get(nct_id, {})
            if clinical_trial:
                checked += 1

                print(clinical_trial.inclusion)
                if len(clinical_trial.inclusion) == 0:
                    inclusions_encoded = model.encode([clinical_trial.text])
                else:
                    inclusions_encoded = model.encode(clinical_trial.inclusion)

                if len(clinical_trial.exclusion) == 0:
                    print('encoding empty')
                    exclusions_encoded = np.zeros(topic_encoded.shape)
                    print(exclusions_encoded.shape)
                else:
                    exclusions_encoded = model.encode(clinical_trial.exclusion)

                topic_inclusion_similarities = cosine_similarity(topic_encoded,
                                                                 inclusions_encoded)  # topic_encoded.dot(inclusions_encoded) / (np.linalg.norm(topic_encoded, axis=1) * np.linalg.norm(inclusions_encoded))
                topic_exclusion_similarities = cosine_similarity(topic_encoded,
                                                                 exclusions_encoded)  # topic_encoded.dot(exclusions_encoded) / (np.linalg.norm(topic_encoded, axis=1) * np.linalg.norm(exclusions_encoded))

                inclusions_similarity[nct_id] = topic_inclusion_similarities
                exclusions_similarity[nct_id] = topic_exclusion_similarities
                combined_score[nct_id] = np.mean(np.sort(topic_inclusion_similarities)[-3:]) * (
                            1 - np.mean(np.sort(topic_exclusion_similarities)[-3:]))

                if np.mean(np.sort(topic_exclusion_similarities)[-2:]) > 0.75:
                    print(f'discarding {nct_id}  --> {topic_no}')
                    excluded_num += 1
                else:
                    included[nct_id] = combined_score[nct_id]

        output_scores[topic_no] = included
        total_checked += checked
        total_excluded += excluded_num
        print(
            f"{topic_no} - {len(included)} - checked: {checked}, excluded {excluded_num}"
        )

        if int(topic_no) % 2 == 0:
            with open(output_file, "w") as fp:
                json.dump(output_scores, fp)

    print(
        f"{total_checked} - {total_excluded} - percentage of excluded {total_excluded / total_checked}%"
    )

    with open(output_file, "w") as fp:
        json.dump(output_scores, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical_trials_folder",
        default="data/external/ClinicalTrials",
        type=str,
        help="path to a folder with clinical trials",
    )
    parser.add_argument(
        "--topic_file",
        default="data/external/topics2021.xml",
        type=str,
        help="path to an xml file with topics data",
    )
    parser.add_argument(
        "--first_stage_file",
        default="data/processed/bm25-baseline-postprocessed.json",
        type=str,
        help="path to a outfile where indexed model will be saved.",
    )
    parser.add_argument(
        "--output_file",
        default="data/processed/bm25-baseline-postprocessed-reranked.json",
        type=str,
        help="path to a outfile where indexed model will be saved.",
    )
    parser.add_argument(
        "--first_n",
        default=2500,
        type=int,
        help="load only first n clinical trial documents (max is ~370k)",
    )

    args = parser.parse_args()

    topics = load_topics_from_xml(args.topic_file)

    cts = parse_clinical_trials_from_folder(
        folder_name=args.clinical_trials_folder, first_n=args.first_n
    )
    cts_dict = {ct.nct_id: ct for ct in cts}

    reranking(
        result_filename=args.first_stage_file,
        output_file=args.output_file,
        clinical_trials_dict=cts_dict,
        topics=topics,
    )
