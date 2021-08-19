import json
import os
import random

import numpy as np
import spacy
from spacy.training.example import Example


def train_entity_recognition_model(
    training_data_file: str, label_file: str, output_dir: str
) -> spacy:
    nlp = spacy.load("en_core_sci_sm")

    # Getting the pipeline component
    ner = nlp.get_pipe("ner")

    with open(training_data_file, "r") as json_file:
        training_data = [json.loads(jline) for jline in json_file.readlines()]
    print(training_data)

    with open(label_file, "r") as json_file:
        labels = json.load(json_file)

    for label in labels:
        print(label["text"])
        ner.add_label(label["text"])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # training data
    TRAIN_DATA = []
    for x in training_data:
        TRAIN_DATA.append((x["data"], {"entities": x["label"]}))

    # TRAINING THE MODEL
    train_loss = []
    with nlp.disable_pipes(*unaffected_pipes):

        # Training for 30 iterations
        for iteration in range(30):
            iter_loss = []

            # shuffling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}

            for batch in spacy.util.minibatch(TRAIN_DATA, size=8):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)

                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=0.5)
                iter_loss.append(losses["ner"])

            train_loss.append(np.mean(iter_loss))
            print(f"iteration: {iteration} - {np.mean(iter_loss)}")

    print(train_loss)
    # Save the  model to directory

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    return nlp


def load_model(model_dir="models/ner_age_gender/"):
    print("Loading from", model_dir)
    return spacy.load(model_dir)


if __name__ == "__main__":
    TRAINING_DATA = "data/raw/d13asfwqUIer121213.jsonl"
    LABEL_FILE = "data/raw/label_config.json"
    OUTPUT_DIR = "models/ner_age_gender/"

    nlp = train_entity_recognition_model(
        training_data_file=TRAINING_DATA, label_file=LABEL_FILE, output_dir=OUTPUT_DIR
    )

    # Testing the model
    doc = nlp(
        """A 58-year-old African-American woman presents to the ER with episodic
        pressing/burning anterior chest pain that began two days earlier for the
        first time in her life. The pain started while she was walking, radiates
        to the back, and is accompanied by nausea, diaphoresis and mild dyspnea,
        but is not increased on inspiration."""
    )
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
