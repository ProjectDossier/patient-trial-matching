import json
from dataclasses import asdict
from typing import List

from tqdm import tqdm

from CTnlp.clinical_trial import ClinicalTrial
from CTnlp.parsers import parse_clinical_trials_from_folder
from trec_cds.features.drug_disease_extraction import EntityExtraction


def convert_trials_to_jsonl(trials: List[ClinicalTrial], outfile: str) -> None:
    """Converts a list of trials to a jsonl file. It also adds the following
    fields to the trial object: "all_criteria", "inclusion_criteria", and
    "exclusion_criteria".
    """
    with open(outfile, "w") as fp:
        for trial in tqdm(trials):
            trial_dict = asdict(trial)
            entities = ee_model.get_entities(trial.criteria)
            trial_dict["all_criteria"] = entities

            inclusion_entities = ee_model.get_entities(" ".join(trial.inclusion))
            trial_dict["inclusion_criteria"] = inclusion_entities

            exclusion_entities = ee_model.get_entities(" ".join(trial.exclusion))
            trial_dict["exclusion_criteria"] = exclusion_entities

            fp.write(json.dumps(trial_dict))
            fp.write("\n")


if __name__ == "__main__":
    INPUT_DATA = "/newstorage4/wkusa/data/trec_cds/ClinicalTrials/"
    OUTFILE = "/newstorage4/wkusa/data/trec_cds/trials_parsed-new.jsonl"

    ee_model = EntityExtraction()

    cts = parse_clinical_trials_from_folder(folder_name=INPUT_DATA)

    convert_trials_to_jsonl(trials=cts, outfile=OUTFILE)
