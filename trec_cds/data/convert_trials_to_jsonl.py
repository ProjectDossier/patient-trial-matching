from CTnlp.parsers import parse_clinical_trials_from_folder
from dataclasses import asdict
import json
from tqdm import tqdm


INPUT_DATA = "/newstorage4/wkusa/data/trec_cds/ClinicalTrials/"

cts = parse_clinical_trials_from_folder(folder_name=INPUT_DATA)

with open("/newstorage4/wkusa/data/trec_cds/trials_parsed.jsonl", 'w') as fp:
    for ct in tqdm(cts):
        fp.write(json.dumps(asdict(ct)) + "\n")
