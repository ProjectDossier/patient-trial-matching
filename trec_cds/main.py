"""Main module contains sample usage of entity recognition methods."""
from dataclasses import asdict
from typing import List

import pandas as pd
from spacy import displacy

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.parsers import (
    load_topics_from_xml,
    parse_clinical_trials_from_folder,
)
from trec_cds.data.topic import Topic
from trec_cds.features.ner import EntityRecognition, get_ner_model, get_displacy_options

if __name__ == "__main__":
    TOPIC_FILE = "data/external/topics2021.xml"
    CLINICAL_TRIALS_FOLDER = "data/external/ClinicalTrials"

    topics: List[Topic] = load_topics_from_xml(TOPIC_FILE)

    er = EntityRecognition()
    er.predict(topics=topics)

    print(topics)

    df = pd.DataFrame([asdict(o) for o in topics])
    df["number"] = df["number"].astype(int)

    df = pd.merge(
        df,
        pd.read_csv("data/raw/topics-healthiness.csv"),
        left_on=["number", "text"],
        right_on=["index", "text"],
    )
    df.to_csv("data/processed/topics.csv", index=False)

    cts: List[ClinicalTrial] = parse_clinical_trials_from_folder(
        folder_name=CLINICAL_TRIALS_FOLDER, first_n=100
    )
    nlp = get_ner_model(custom_ner_model_path="models/ner_age_gender-new")

    docs = []
    for clinical_trial in cts:
        doc = nlp(clinical_trial.criteria)
        docs.append(doc)

    options = get_displacy_options()

    displacy.serve(docs, style="ent", options=options)

    # print(cts)
