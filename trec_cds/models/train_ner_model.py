import json
import random
from pathlib import Path

import numpy as np
import spacy
from spacy.training.example import Example


def train_age_gender_model() -> spacy:
    nlp = spacy.load("en_core_web_sm")

    # Getting the pipeline component
    ner = nlp.get_pipe("ner")

    with open("data/raw/d13asfwqUIer121213.jsonl", "r") as json_file:
        training_data = [json.loads(jline) for jline in json_file.readlines()]

    print(training_data)

    with open("data/raw/label_config.json", "r") as json_file:
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
        for iteration in range(23):
            iter_loss = []

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}

            for batch in spacy.util.minibatch(TRAIN_DATA, size=8):
                for text, annotations in batch:
                    # create Example
                    doc = nlp.make_doc(text)

                    # print(
                    #     spacy.training.offsets_to_biluo_tags(
                    #         nlp.make_doc(text), annotations
                    #     )
                    # )
                    example = Example.from_dict(doc, annotations)
                    # Update the model
                    nlp.update([example], losses=losses, drop=0.5)
                # print("Losses", losses)
                iter_loss.append(losses["ner"])

            train_loss.append(np.mean(iter_loss))
            print(f"iteration: {iteration} - {np.mean(iter_loss)}")

    print(train_loss)
    # Save the  model to directory
    output_dir = Path("models/ner_age_gender/")
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    return nlp


def load_model(model_dir="models/ner_age_gender/"):
    print("Loading from", model_dir)
    return spacy.load(model_dir)


if __name__ == "__main__":
    nlp = train_age_gender_model()

    # Testing the model
    doc = nlp(
        """A 58-year-old African-American woman presents to the ER with episodic
        pressing/burning anterior chest pain that began two days earlier for the
        first time in her life. The pain started while she was walking, radiates
        to the back, and is accompanied by nausea, diaphoresis and mild dyspnea,
        but is not increased on inspiration."""
    )
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
