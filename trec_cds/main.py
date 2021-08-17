import pandas as pd
from spacy import displacy

from trec_cds.data.parsers import (
    load_topics_from_xml,
    parse_clinical_trials_from_folder,
)
from trec_cds.features.ner import EntityRecognition, get_ner_model, get_displacy_options

if __name__ == "__main__":
    topic_file = "data/external/topics2021.xml"
    clinical_trials_folder = "data/external/ClinicalTrials"

    topics = load_topics_from_xml(topic_file)

    er = EntityRecognition()
    er.predict(topics=topics)

    print(topics)

    df = pd.DataFrame([o.__dict__ for o in topics])
    df["number"] = df["number"].astype(int)

    df = pd.merge(
        df,
        pd.read_csv("data/raw/topics-healthiness.csv"),
        left_on=["number", "text"],
        right_on=["index", "text"],
    )
    df.to_csv("data/processed/topics.csv", index=False)

    cts = parse_clinical_trials_from_folder(
        folder_name=clinical_trials_folder, first_n=100
    )
    nlp = get_ner_model(custom_ner_model_path="models/ner_age_gender-new")

    docs = []
    for topic in cts:
        doc = nlp(topic.criteria)
        docs.append(doc)

    options = get_displacy_options()

    displacy.serve(docs, style="ent", options=options)

    # print(cts)
