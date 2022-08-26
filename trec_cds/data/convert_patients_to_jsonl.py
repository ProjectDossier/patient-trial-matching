import json



import json
from dataclasses import asdict
from typing import List

from CTnlp.patient import load_patients_from_xml
from CTnlp.patient import Patient
from CTnlp.patient import load_patients_from_xml
from trec_cds.features.drug_disease_extraction import EntityExtraction
from trec_cds.features.entity_recognition import EntityRecognition, normalise_smoking, normalise_drinking

ee_model = EntityExtraction()
er = EntityRecognition()


def convert_patients_to_jsonl(patients: List[Patient], outfile):
    with open(outfile, "w") as fp:
        for patient in patients:
            patient_dict = asdict(patient)
            entities = ee_model.get_entities(patient.description)
            patient_dict.update(entities)
            patient_dict['is_smoker'] = normalise_smoking(patient_dict['negated_entities'], patient_dict['positive_entities'])
            patient_dict['is_drinker'] = normalise_drinking(patient_dict['negated_entities'], patient_dict['positive_entities'])

            fp.write(json.dumps(patient_dict))
            fp.write("\n")


if __name__ == "__main__":

    for in_file in ['topics2021', 'topics2022']:
        patients = load_patients_from_xml(f"data/external/{in_file}.xml")
        er.predict(topics=patients)
        OUTFILE = f"data/external/{in_file}.jsonl"

        convert_patients_to_jsonl(patients=patients, outfile=OUTFILE)

