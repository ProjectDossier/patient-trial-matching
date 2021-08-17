import logging

import pandas as pd
import spacy
from tqdm import tqdm

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import parse_clinical_trials_from_folder


class ClinicalTrialsFeatures:
    def __init__(self):
        self.nlp = spacy.load(
            "en_core_sci_sm",
            disable=[
                "ner",
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        logging.warning("loaded spacy language model for preprocessing Clinical Trials")

    def get_text(self, clinical_trial: ClinicalTrial):
        text = (
                clinical_trial.brief_title
                + clinical_trial.official_title
                + clinical_trial.summary
                + clinical_trial.criteria
        )
        clinical_trial.text = text

    def preprocess_text(self, clinical_trial: ClinicalTrial):
        preprocessed = self.nlp(clinical_trial.text)

        clinical_trial.text_preprocessed = [
            token.text for token in preprocessed if not token.is_stop
        ]


if __name__ == "__main__":
    feature_builder = ClinicalTrialsFeatures()

    clinical_trials_folder = "data/external/ClinicalTrials"
    cts = parse_clinical_trials_from_folder(folder_name=clinical_trials_folder)

    for ct in tqdm(cts):
        feature_builder.get_text(clinical_trial=ct)
        feature_builder.preprocess_text(clinical_trial=ct)
        # print(ct.text_preprocessed)

    df = pd.DataFrame([o.__dict__ for o in cts])
    df.to_csv("data/processed/clinical_trials.csv", index=False)
