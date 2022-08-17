import json
import logging
from typing import Dict, List

import pandas as pd

from CTnlp.clinical_trial import ClinicalTrial
from CTnlp.parsers import parse_clinical_trials_from_folder
from CTnlp.patient.parser import load_patients_from_xml
from CTnlp.patient.patient import Patient
from CTnlp.utils import Gender
from trec_cds.features.entity_recognition import EntityRecognition


def postprocessing(
    result_filename: str,
    output_file: str,
    clinical_trials_dict: Dict[str, ClinicalTrial],
    topics: List[Patient],
):
    """Post processes result file by removing gender, age and health status mismatches
    between topic and clinical trial data."""
    with open(result_filename) as fp:
        results = json.load(fp)

    total_checked = 0
    total_excluded = 0

    output_scores = {}
    for topic_no in results:
        included = {}
        logging.info(topic_no)
        excluded_num = 0
        checked = 0

        for nct_id, score in results[topic_no].items():
            healthy = topics[int(topic_no) - 1].is_healthy
            gender = topics[int(topic_no) - 1].gender
            age = topics[int(topic_no) - 1].age

            clinical_trial = clinical_trials_dict.get(nct_id, {})
            if clinical_trial:
                checked += 1
                if clinical_trial.gender != gender and clinical_trial.gender not in [
                    Gender.all,
                    Gender.unknown,
                ]:
                    logging.info("gender mismatch")
                    excluded_num += 1
                    continue

                if (
                    age != -1
                    and clinical_trial.minimum_age
                    and age < clinical_trial.minimum_age
                ):
                    logging.info("skipping because of minimum_age age")
                    excluded_num += 1
                    continue
                if (
                    age != -1
                    and clinical_trial.maximum_age
                    and age > clinical_trial.maximum_age
                ):
                    logging.info("skipping because of maximum_age age")
                    excluded_num += 1
                    continue

                if healthy and not clinical_trial.accepts_healthy_volunteers:
                    logging.info("trial not accepting healthy volunteers")
                    excluded_num += 1
                    continue

            included[nct_id] = score

        output_scores[topic_no] = included
        total_checked += checked
        total_excluded += excluded_num
        print(f"{topic_no=} - {len(included)=} - {checked=}, excluded {excluded_num=}")

    print(
        f"{total_checked=} - {total_excluded=} - \
        percentage of excluded {total_excluded / total_checked}%"
    )

    with open(output_file, "w") as fp:
        json.dump(output_scores, fp)


if __name__ == "__main__":
    topic_file = "data/external/topics2021.xml"
    clinical_trials_folder = "data/external/ClinicalTrials"
    first_stage_results_file = "data/processed/bm25-baseline-scores-4000.json"

    topics = load_patients_from_xml(patient_file=topic_file)
    er = EntityRecognition()
    er.predict(topics=topics)

    health_status_df = pd.read_csv("data/raw/topics-healthiness.csv")
    for topic in topics:
        label = health_status_df[topic.patient_id == health_status_df["index"]][
            "label"
        ].tolist()[0]
        if label == "HEALTHY":
            topic.healthy = True
        else:
            topic.healthy = False

    cts = parse_clinical_trials_from_folder(
        folder_name=clinical_trials_folder, first_n=400000
    )

    cts_dict = {ct.nct_id: ct for ct in cts}

    postprocessing(
        result_filename=first_stage_results_file,
        output_file="data/processed/bm25-baseline-postprocessed.json",
        clinical_trials_dict=cts_dict,
        topics=topics,
    )
