from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.clinical_trial import ClinicalTrial
from dataclasses import asdict
import json
from tqdm import tqdm
from trec_cds.features.drug_disease_extraction import EntityExtraction
from typing import List

ee_model = EntityExtraction()

def convert_trials_to_jsonl(trials: List[ClinicalTrial], outfile):
    with open(outfile, "w") as fp:
        for trial in tqdm(trials):
            trial_dict = asdict(trial)
            entities = ee_model.get_entities(trial.criteria)
            trial_dict['all_criteria'] = entities

            inclusion_entities = ee_model.get_entities(" ".join(trial.inclusion))
            trial_dict['inclusion_criteria'] = inclusion_entities

            exclusion_entities = ee_model.get_entities(" ".join(trial.exclusion))
            trial_dict['exclusion_criteria'] = exclusion_entities

            fp.write(json.dumps(trial_dict))
            fp.write("\n")


INPUT_DATA = "/newstorage4/wkusa/data/trec_cds/ClinicalTrials/"

cts = parse_clinical_trials_from_folder(folder_name=INPUT_DATA)

OUTFILE = "/newstorage4/wkusa/data/trec_cds/trials_parsed-new.jsonl"
convert_trials_to_jsonl(trials=cts, outfile=OUTFILE)
