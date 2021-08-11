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

                    # print(spacy.training.offsets_to_biluo_tags(nlp.make_doc(text), annotations))
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
        """
                <description>A 58-year-old African-American woman presents to the ER with episodic pressing/burning 
                anterior chest pain that began two days earlier for the first time in her life. The pain started while 
                she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. The latest episode of pain ended half an hour prior to her arrival. She is known to have hypertension and obesity. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. She currently takes no medications. Physical examination is normal. The EKG shows nonspecific changes.</description>
      <summary> 58-year-old woman with hypertension and obesity presents with exercise-related episodic chest pain radiating to the back.</summary>
     </topic>
     <topic number="2" type="diagnosis">
      <description> An 8-year-old male presents in March to the ER with fever up to 39 C, dyspnea and cough for 2 days. He has just returned from a 5 day vacation in Colorado. Parents report that prior to the onset of fever and cough, he had loose stools. He denies upper respiratory tract symptoms. On examination he is in respiratory distress and has bronchial respiratory sounds on the left. A chest x-ray shows bilateral lung infiltrates.</description>
      <summary> 8-year-old boy with 2 days of loose stools, fever, and cough after returning from a trip to Colorado. 8-year-old M Chest x-ray shows bilateral lung infiltrates.</summary>
    19 yo Hispanic female is a 16-year-old girl
    The patient is a 3-day-old female infant with jaundice that started one day ago. She was born at 34w of gestation and kept in an incubator due to her gestational age. Vital signs were reported as: axillary temperature: 36.3°C, heart rate: 154 beats/min, respiratory rate: 37 breaths/min, and blood pressure: 65/33 mm Hg. Her weight is 2.1 kg, length is 45 cm, and head circumference 32 cm. She presents with yellow sclera and icteric body. Her liver and spleen are normal to palpation.
    79 yo F with multifactorial chronic hypoxemia and dyspnea thought due to diastolic CHF, pulmonary hypertension thought secondary to a chronic ASD and COPD on 5L home oxygen admitted with complaints of worsening shortness of breath. Cardiology consult recommended a right heart cath for evaluation of response to sildenafil but the patient refused. Pulmonary consult recommended an empiric, compassionate sildenafil trial due to severe dyspneic symptomology preventing outpatient living, and the patient tolerated an inpatient trial without hypotension. Patient to f/u with pulmonology to start sildenifil chronically as outpatient as prior authorization is obtained.
    
              """
    )
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
