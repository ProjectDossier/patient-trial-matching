import logging

import spacy

from trec_cds.data.clinical_trial import ClinicalTrial


class ClinicalTrialsFeatures:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_scibert")
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
        self.nlp(clinical_trial.text)


if __name__ == "__main__":
    feature_builder = ClinicalTrialsFeatures()
