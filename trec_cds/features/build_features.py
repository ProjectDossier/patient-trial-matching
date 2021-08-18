import logging
from dataclasses import asdict

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

    def preprocess_text(self, clinical_trial: ClinicalTrial):
        preprocessed = self.nlp(clinical_trial.text)

        clinical_trial.text_preprocessed = [
            token.text for token in preprocessed if not token.is_stop
        ]


if __name__ == "__main__":
    feature_builder = ClinicalTrialsFeatures()

    CLINICAL_TRIALS_FOLDER = "data/external/ClinicalTrials"
    cts = parse_clinical_trials_from_folder(folder_name=CLINICAL_TRIALS_FOLDER, first_n=100)

    for ct in tqdm(cts):
        feature_builder.preprocess_text(clinical_trial=ct)
        # print(ct.text_preprocessed)

    df = pd.DataFrame([asdict(o) for o in cts])
    df.to_csv("data/processed/clinical_trials.csv", index=False)
