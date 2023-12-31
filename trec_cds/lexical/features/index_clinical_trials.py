import argparse
import json
import pickle
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus
from tqdm import tqdm

from CTnlp.clinical_trial import ClinicalTrial
from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.patient import load_patients_from_xml
from CTnlp.patient import Patient
from trec_cds.lexical.features.build_features import ClinicalTrialsFeatures


class Indexer:
    """Wrapper around BM25Okapi class that indexes ClinicalTrials and allows for
    querying them with Topic data. input data must be preprocessed and tokenized."""

    index: BM25Plus
    lookup_table: Dict[int, str]

    def index_clinical_trials(self, clinical_trials: List[ClinicalTrial]):
        cts_tokenized = []

        for _clinical_trial in tqdm(clinical_trials):
            cts_tokenized.append(_clinical_trial.text_preprocessed)

        self.index = BM25Plus(cts_tokenized)
        self.lookup_table = {x_index: x.nct_id for x_index, x in enumerate(clinical_trials)}

    def index_text(self, text, lookup_table):
        self.index = BM25Plus(text)
        self.lookup_table = lookup_table

    def query_single(self, query: List[str], return_top_n: int) -> Dict[str, float]:
        """Query needs to be tokenized."""
        topic_scores = {}
        doc_scores = self.index.get_scores(query)
        for index, score in zip(
            np.argsort(doc_scores)[-return_top_n:], np.sort(doc_scores)[-return_top_n:]
        ):
            topic_scores[self.lookup_table[index]] = score

        return topic_scores

    def load_index(self, filename: str):
        """Loads index from a pickled file into index variable."""
        with open(filename, "rb") as _fp:
            self.index = pickle.load(_fp)

    def save_index(self, filename: str):
        """Saves index into a pickled file."""
        with open(filename, "wb") as _fp:
            pickle.dump(self.index, _fp)


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
        "--model_outfile",
        default="/newstorage4/wkusa/data/trec_cds/models/bm25-baseline1.p",
        type=str,
        help="path to an outfile where indexed model will be saved.",
    )
    parser.add_argument(
        "--results_outfile",
        default="/newstorage4/wkusa/data/trec_cds/data/processed/bm25p-baseline-220824.json",
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
        folder_name=args.clinical_trials_folder, first_n=args.first_n
    )

    feature_builder = ClinicalTrialsFeatures(spacy_language_model_name='en_core_sci_lg')
    for clinical_trial in tqdm(cts):
        feature_builder.preprocess_clinical_trial(clinical_trial=clinical_trial)

    indexer = Indexer()
    indexer.index_clinical_trials(clinical_trials=cts)
    indexer.save_index(filename=args.model_outfile)

    topics: List[Patient] = load_patients_from_xml(patient_file=args.topic_file)

    output_scores = {}
    for topic in tqdm(topics):
        doc = feature_builder.preprocess_text(topic.description)
        output_scores[topic.patient_id] = indexer.query_single(
            query=doc, return_top_n=args.return_top_n
        )

    with open(args.results_outfile, "w") as fp:
        json.dump(output_scores, fp)
