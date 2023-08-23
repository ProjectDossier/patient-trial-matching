import argparse
import json
from dataclasses import asdict
from typing import List

from CTnlp.patient import Patient
from CTnlp.patient import load_patients_from_xml

from trec_cds.features.drug_disease_extraction import EntityExtraction
from trec_cds.features.entity_recognition import (
    EntityRecognition,
    normalise_smoking,
    normalise_drinking,
)


def convert_patients_to_jsonl(patients: List[Patient], outfile: str) -> None:
    """Converts a list of patients to a jsonl file. Each line is a json
    representation of a patient. This function adds the following fields
    to the patient object: "is_smoker", "is_drinker", "entities",
    "negated_entities", "positive_entities".

    :param patients:
    :param outfile:
    :return:
    """
    with open(outfile, "w") as fp:
        for patient in patients:
            patient_dict = asdict(patient)
            entities = ee_model.get_entities(patient.description)
            patient_dict.update(entities)
            patient_dict["is_smoker"] = normalise_smoking(
                patient_dict["cmh_entities"]
            )
            patient_dict["is_drinker"] = normalise_drinking(
                patient_dict["cmh_entities"]
            )

            fp.write(json.dumps(patient_dict))
            fp.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        default="data/external/",
        help="Path to the folder containing the clinical trials.",
    )
    args = parser.parse_args()

    ee_model = EntityExtraction()
    er = EntityRecognition()

    for in_file in ["topics2021", "topics2022", "topics2023"]:
        patients = load_patients_from_xml(
            f"{args.input_data}/{in_file}.xml", input_type="TREC"
        )
        er.predict(topics=patients)
        OUTFILE = f"{args.input_data}/{in_file}.jsonl"

        convert_patients_to_jsonl(patients=patients, outfile=OUTFILE)
