import os
import xml.etree.ElementTree as ET
from glob import glob
from typing import List, Union

from trec_cds.data.clinical_trial import ClinicalTrial
from trec_cds.data.topic import Topic


def parse_topics_from_xml(topic_file: str) -> List[Topic]:
    """Parses topics from single XML file and creates a Topic class instance for each parsed item.

    :param topic_file: str
    :return: List of topics
    """
    tree = ET.parse(topic_file)
    root = tree.getroot()

    topics = []
    for elem in root:
        topics.append(Topic(number=elem.attrib["number"], text=elem.text))

    return topics


def parse_clinical_trials_from_folder(
    folder_name: str, first_n: Union[None, int] = None
) -> List[ClinicalTrial]:
    files = [y for x in os.walk(folder_name) for y in glob(os.path.join(x[0], "*.xml"))]

    if first_n:
        files = files[:first_n]

    clinical_trials = []
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()

        org_study_id = root.find("id_info").find("org_study_id").text

        eligibility = root.find("eligibility")
        clinical_trials.append(
            ClinicalTrial(
                org_study_id=org_study_id,
                summary=root.find("brief_summary")[0].text,
                description=root.find("detailed_description")[0].text,
                criteria=eligibility.find("criteria")[0].text,
                gender=eligibility.find("gender").text,
                minimum_age=eligibility.find("minimum_age").text,
                maximum_age=eligibility.find("maximum_age").text,
                healthy_volunteers=eligibility.find("healthy_volunteers").text,
            )
        )

    return clinical_trials


if __name__ == "__main__":
    topic_file = "../data/external/topics2021.xml"
    folder_name = "../data/external/ClinicalTrials"

    topics = parse_topics_from_xml(topic_file)

    print(topics)

    cts = parse_clinical_trials_from_folder(folder_name=folder_name)
