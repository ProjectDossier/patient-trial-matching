import json
from dataclasses import asdict
from typing import List

from CTnlp.patient import Patient
from CTnlp.patient import load_patients_from_xml
from trec_cds.features.drug_disease_extraction import EntityExtraction

ee_model = EntityExtraction()


def convert_patients_to_jsonl(patients: List[Patient], outfile):
    with open(outfile, "w") as fp:
        for patient in patients:
            patient_dict = asdict(patient)
            entities = ee_model.get_entities(patient.description)
            patient_dict.update(entities)
            fp.write(json.dumps(patient_dict))
            fp.write("\n")


if __name__ == "__main__":
    patients = []
    patients.extend(
        load_patients_from_xml("../../data/external/topics2014.xml", input_type="CSIRO")
    )
    patients.extend(load_patients_from_xml("../../data/external/topics2021.xml"))
    patients.extend(load_patients_from_xml("../../data/external/topics2022.xml"))

    OUTFILE = "../../data/external/patients_2.jsonl"

    convert_patients_to_jsonl(patients=patients, outfile=OUTFILE)
