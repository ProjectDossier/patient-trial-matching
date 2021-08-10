import json
import random
from pathlib import Path
import re

import spacy
from spacy import displacy

from trec_cds.data.parsers import parse_topics_from_xml
from trec_cds.data.utils import Gender


def get_displacy_params():
    with open("data/raw/label_config.json", "r") as json_file:
        labels = json.load(json_file)

    colors = {
        label[
            "text"
        ]: f"{'#' + ''.join([random.choice('3456789ABCDEF') for i in range(6)])}"
        for label in labels
    }

    options = {"ents": [label["text"] for label in labels], "colors": colors}
    return options


def get_ner_model():
    nlp = spacy.load("en_ner_bc5cdr_md")

    model_dir = "models/ner_age_gender/"
    age_gender_nlp = spacy.load(model_dir)

    nlp.add_pipe(
        "ner",
        name="ner_age_gender",
        source=age_gender_nlp,
        after="ner",
    )

    return nlp


def extract_age(text):
    match = re.search(r"\d{1,2}", text)
    if match is not None:
        return match.group(0)
    else:
        return None


def extract_gender(text):
    male = ['M', 'male', 'boy', 'man', 'gentleman']
    female = ['F', 'female', 'girl', 'woman']

    for pattern in male:
        match = re.search(rf"\b{pattern}\b", text)
        if match is not None:
            return Gender.male

    for pattern in female:
        match = re.search(rf"\b{pattern}\b", text)
        if match is not None:
            return Gender.female

    return Gender.unknown


if __name__ == "__main__":
    topic_file = "data/external/topics2021.xml"
    topics = parse_topics_from_xml(topic_file)

    nlp = get_ner_model()

    docs = []
    for topic in topics:
        doc = nlp(topic.text)
        docs.append(doc)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

    options = get_displacy_params()

    # displacy.serve(docs, style="ent")
    displacy.serve(docs, style="ent", options=options)

    svg = displacy.render(docs, style="ent", options=options)
    output_path = Path("reports/figures/ner.svg")
    output_path.open("w", encoding="utf-8").write(svg)
