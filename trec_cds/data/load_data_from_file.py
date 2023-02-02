import json
from typing import List, Dict, Any


def load_jsonl(infile: str) -> List[Dict[str, Any]]:
    with open(infile) as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == "__main__":
    patient_file = "topics2021"
    infile = f"data/processed/{patient_file}.jsonl"

    patients = load_jsonl(infile)
    print([patient["is_smoker"] for patient in patients])
    print([patient["is_drinker"] for patient in patients])
