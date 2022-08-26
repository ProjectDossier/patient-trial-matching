import json
import random
import re
from pathlib import Path
from typing import List, Dict, Union

import spacy
from spacy import displacy

from CTnlp.patient import load_patients_from_xml
from CTnlp.patient import Patient
from CTnlp.utils import Gender


def normalise_smoking(negated_entities, positive_entities):
    if "smoke" in negated_entities:
        return False

    if "smoke" in positive_entities:
        return True
    return None

def normalise_drinking(negated_entities, positive_entities):
    if "alcohol" in negated_entities:
        return False

    if "alcohol" in positive_entities:
        return True
    return None



def get_displacy_options(
    label_file: str = "data/raw/label_config.json",
) -> Dict[str, List]:
    """Loads labels from a label file and creates a options dict with labels
    colours that will be used by spacy.displacy

    :param label_file: path to a file containing NER labels
    :return:
    """
    with open(label_file, "r") as json_file:
        labels = json.load(json_file)

    colors = {
        label[
            "text"
        ]: f"{'#' + ''.join([random.choice('3456789ABCDEF') for _ in range(6)])}"
        for label in labels
    }  # random.choice starting from 3 as we don't want too dark labels

    return {"ents": [label["text"] for label in labels], "colors": colors}


def get_ner_model(
    custom_ner_model_path: str = "models/ner_age_gender/",
) -> spacy.Language:
    """Load Named Entity Recognition spacy model that combines pre-trained
    en_ner_bc5cdr_md for Disease and Chemical
    prediction with a custom model trained on topics corpora.

    :return: spacy NER model
    """
    base_nlp = spacy.load("en_ner_bc5cdr_md")
    age_gender_nlp = spacy.load(custom_ner_model_path)

    base_nlp.add_pipe(
        "ner",
        name=custom_ner_model_path,
        source=age_gender_nlp,
        after="ner",
    )

    return base_nlp


class EntityRecognition:
    def __init__(self, custom_ner_model_path: str = "models/ner_age_gender/"):
        self.nlp = get_ner_model(custom_ner_model_path=custom_ner_model_path)
        print("loaded spacy language model for entity detection")

    def predict(self, topics: List[Patient]):
        for topic in topics:
            doc = self.nlp(topic.description)

            age_entities = [ent.text for ent in doc.ents if ent.label_ == "AGE"]
            if len(age_entities) > 0:
                topic.age = self.extract_age_from_entity(age_entities[0])
            else:
                topic.age = -1

            gender_entities = [ent.text for ent in doc.ents if ent.label_ == "GENDER"]
            if len(gender_entities) > 0:
                topic.gender = self.extract_gender_from_entity(gender_entities[0])
            else:
                topic.gender = self.extract_gender_from_text(topic.description)

        return topics

    @staticmethod
    def extract_age_from_entity(text: str) -> Union[int, float, None]:
        """Extracts age from candidate string coming from AGE entity in spacy NER model
        :param text: string containing entity containing age candidate
        :return: int: patient's age. If integer is not found, functions returns None
        """
        match = re.search(r"(\d{1,2})[- ](month[s]?[- ]old)", text)
        if match is not None:
            return int(match.group(1)) / 12

        match = re.search(r"(\d{1,2})[- ](day[s]?[- ]old)", text)
        if match is not None:
            return int(match.group(1)) / 365

        match = re.search(r"\d{1,2}", text)
        if match is not None:
            return int(match.group(0))

        return None

    @staticmethod
    def extract_gender_from_entity(text: str) -> Gender:
        """Extracts gender from candidate string coming from GENDER entity in
         spacy NER model.

        :param text: string containing entity containing gender candidate
        :return: Gender
        """
        male = ["M", "male", "boy", "man", "gentleman"]
        female = ["F", "female", "girl", "woman", "lady"]

        for pattern in male:
            match = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
            if match is not None:
                return Gender.male

        for pattern in female:
            match = re.search(rf"\b{pattern}\b", text, re.IGNORECASE)
            if match is not None:
                return Gender.female

        return Gender.unknown

    @staticmethod
    def extract_gender_from_text(text: str) -> Gender:
        """Simple heuristic to estimate the gender based on count of a female/male
        pronouns in a text.

        :param text:
        :return:
        """
        male_pronoun = "he"
        female_pronoun = "she"

        male_matches = re.findall(rf"\b{male_pronoun}\b", text, re.IGNORECASE)
        female_matches = re.findall(rf"\b{female_pronoun}\b", text, re.IGNORECASE)

        if male_matches > female_matches:
            return Gender.male
        elif male_matches < female_matches:
            return Gender.female
        else:
            return Gender.unknown


if __name__ == "__main__":
    TOPIC_FILE = "data/external/topics2021.xml"
    topics = load_patients_from_xml(patient_file=TOPIC_FILE)

    nlp = get_ner_model(custom_ner_model_path="models/ner_age_gender-new")

    docs = []
    for topic in topics:
        doc = nlp(topic.description)
        docs.append(doc)

    options = get_displacy_options()

    displacy.serve(docs, style="ent", options=options)

    svg = displacy.render(docs, style="ent", options=options)
    output_path = Path("reports/figures/ner.svg")
    output_path.open("w", encoding="utf-8").write(svg)
