import logging
from dataclasses import asdict

import pandas as pd
import spacy
from tqdm import tqdm

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import parse_clinical_trials_from_folder


class ClinicalTrialsFeatures:
    def __init__(self, spacy_language_model: str = "en_core_sci_sm"):
        self.nlp = spacy.load(
            spacy_language_model,
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

    def preprocess_text(self, clinical_trial: ClinicalTrial) -> None:
        """Preprocesses a clinical trial text field using spacy tokenizer and removing
        stopwords. Preprocessed text is saved to a variable in the clinical_trial
        object."""
        preprocessed = self.nlp(clinical_trial.text)

        clinical_trial.text_preprocessed = [
            token.text for token in preprocessed if not token.is_stop
        ]


if __name__ == "__main__":
    CLINICAL_TRIALS_FOLDER = "data/external/ClinicalTrials"
    FIRST_N = 2000
    OUTPUT_FILE = "data/processed/clinical_trials.csv"

    cts = parse_clinical_trials_from_folder(
        folder_name=CLINICAL_TRIALS_FOLDER, first_n=FIRST_N
    )

    feature_builder = ClinicalTrialsFeatures()
    for ct in tqdm(cts):
        feature_builder.preprocess_text(clinical_trial=ct)

    df = pd.DataFrame([asdict(ct) for ct in cts])
    df.to_csv(OUTPUT_FILE, index=False)
