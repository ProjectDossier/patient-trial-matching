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

        org_study_id = getattr(root.find("id_info").find("org_study_id"), "text", None)

        brief_summary = root.find("brief_summary")
        if brief_summary:
            brief_summary = brief_summary[0].text

        description = root.find("detailed_description")
        if description:
            description = description[0].text

        eligibility = root.find("eligibility")
        if not eligibility:
            criteria = None
            gender = None
            minimum_age = None
            maximum_age = None
            healthy_volunteers = None
        else:
            criteria = eligibility.find("criteria")
            if criteria:
                criteria = criteria[0].text

            gender = getattr(eligibility.find("gender"), "text", None)
            minimum_age = getattr(eligibility.find("minimum_age"), "text", None)
            maximum_age = getattr(eligibility.find("maximum_age"), "text", None)
            healthy_volunteers = getattr(
                eligibility.find("healthy_volunteers"), "text", None
            )

        try:
            clinical_trials.append(
                ClinicalTrial(
                    org_study_id=org_study_id,
                    summary=brief_summary,
                    description=description,
                    criteria=criteria,
                    gender=gender,
                    minimum_age=minimum_age,
                    maximum_age=maximum_age,
                    healthy_volunteers=healthy_volunteers,
                )
            )
        except Exception as E:
            print(file, E)

    return clinical_trials
