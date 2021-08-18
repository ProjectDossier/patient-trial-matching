import argparse
import json
import pickle
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import (
    parse_clinical_trials_from_folder,
    load_topics_from_xml,
)
from trec_cds.features.build_features import ClinicalTrialsFeatures


class Indexer:
    index: BM25Okapi

    def __init__(self):
        pass

    def index_clinical_trials(self, clinical_trials: List[ClinicalTrial]):
        cts_tokenized = []

        for _clinical_trial in tqdm(clinical_trials):
            cts_tokenized.append(_clinical_trial.text_preprocessed)

        self.index = BM25Okapi(cts_tokenized)

    def query(self, docs: List[List[str]]):
        pass

    def query_single(self, query: List[str], return_top_n: int) -> Dict[str, float]:
        topic_scores = {}
        doc_scores = self.index.get_scores(query)
        for index, score in zip(
                np.argsort(doc_scores)[-return_top_n:], np.sort(doc_scores)[-return_top_n:]
        ):
            topic_scores[cts[index].nct_id] = score

        return topic_scores

    def load_index(self, filename: str):
        self.index = pickle.load(open(filename, "rb"))

    def save_index(self, filename: str):
        pickle.dump(self.index, open(filename, "wb"))


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
        "--model_outfile",
        default="models/bm25-baseline1.p",
        type=str,
        help="path to a outfile where indexed model will be saved.",
    )
    parser.add_argument(
        "--first_n",
        default=250000,
        type=int,
        help="load only first n clinical trial documents (max is ~370k)",
    )

    parser.add_argument(
        "--return_top_n",
        default=2000,
        type=int,
        help="return top n results from retrieval model",
    )

    args = parser.parse_args()

    cts = parse_clinical_trials_from_folder(
        folder_name=args.clinical_trials_folder, first_n=args.first_n
    )

    feature_builder = ClinicalTrialsFeatures()
    for clinical_trial in tqdm(cts):
        feature_builder.preprocess_clinical_trial(clinical_trial=clinical_trial)

    indexer = Indexer()
    indexer.index_clinical_trials(clinical_trials=cts)
    indexer.save_index(filename=args.model_outfile)

    topics = load_topics_from_xml(topic_file=args.topic_file)

    output_scores = {}
    for topic in tqdm(topics):
        doc = feature_builder.preprocess_text(topic.text)
        topic_scores = indexer.query_single(query=doc, return_top_n=args.return_top_n)

        output_scores[topic.number] = topic_scores

    with open("data/processed/bm25-baseline-scores.json", "w") as fp:
        json.dump(output_scores, fp)
