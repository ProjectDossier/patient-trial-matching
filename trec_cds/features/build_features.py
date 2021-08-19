"""Module containing class that preprocesses ClinicalTrial objects and builds features
for model predictions."""
import logging
from dataclasses import asdict
from typing import List

import pandas as pd
import spacy
from tqdm import tqdm

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import parse_clinical_trials_from_folder


class ClinicalTrialsFeatures:
    """Class wrapping nlp spacy language model that tokenizes and removes stopwords
    for ClinicalTrial objects"""

    def __init__(self, spacy_language_model_name: str = "en_core_sci_sm"):
        self.nlp = spacy.load(
            spacy_language_model_name,
            disable=[
                "ner",
                "tok2vec",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
            ],
        )
        logging.info("loaded spacy language model for preprocessing Clinical Trials")

    def preprocess_clinical_trial(self, clinical_trial: ClinicalTrial) -> None:
        """Preprocesses a clinical trial text field using spacy tokenizer and removing
        stopwords. Preprocessed text is saved to a variable in the clinical_trial
        object."""
        preprocessed = self.nlp(clinical_trial.text)

        clinical_trial.text_preprocessed = [
            token.text for token in preprocessed if not token.is_stop
        ]

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocesses a custom text field using spacy tokenizer and removing
        stopwords. Preprocessed text is returned as a List of tokenized strings.

        This method can be used to obtain the same preprocessing for e.g. Topic data
        as for ClinicalTrial."""
        preprocessed = self.nlp(text)
        return [token.text for token in preprocessed if not token.is_stop]


if __name__ == "__main__":
    CLINICAL_TRIALS_FOLDER = "data/external/ClinicalTrials"
    FIRST_N = 2000
    OUTPUT_FILE = "data/processed/clinical_trials.csv"

    cts = parse_clinical_trials_from_folder(
        folder_name=CLINICAL_TRIALS_FOLDER, first_n=FIRST_N
    )

    feature_builder = ClinicalTrialsFeatures()
    for clinical_trial in tqdm(cts):
        feature_builder.preprocess_clinical_trial(clinical_trial=clinical_trial)

    df = pd.DataFrame([asdict(ct) for ct in cts])
    df.to_csv(OUTPUT_FILE, index=False)
