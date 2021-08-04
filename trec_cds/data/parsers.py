import os
import re
import xml.etree.ElementTree as ET
from glob import glob
from typing import List, Union, Tuple

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


def parse_criteria(criteria: str) -> Union[None, Tuple[List[str], List[str]]]:
    exclusion = ""

    inclusion_criteria_strings = [
        "Inclusion Criteria",
        "Inclusion criteria",
        "Inclusive criteria",
        "INCLUSION CRITERIA",
    ]
    for inclusion_criteria_string in inclusion_criteria_strings:
        if criteria.find(inclusion_criteria_string) != -1:
            criteria_after_split = criteria.split(inclusion_criteria_string)
            break
        else:
            criteria_after_split = criteria

    if len(criteria_after_split) == 2:
        empty, tmp_inclusion = criteria_after_split
    elif len(criteria_after_split) == 1:
        return None
    else:
        return None

    # if empty.strip() != "":
    #     print('empty', empty)

    exclusion_criteria_strings = [
        "Exclusion Criteria",
        "Exclusion criteria",
        "Exclusive criteria",
        "EXCLUSION CRITERIA",
        "ECLUSION CRITERIA",
        "EXCLUSION CRITIERIA",
    ]
    for exclusion_criteria_string in exclusion_criteria_strings:
        if tmp_inclusion.find(exclusion_criteria_string) != -1:
            inclusion_exclusion_split = tmp_inclusion.split(exclusion_criteria_string)
            break
        else:
            inclusion_exclusion_split = [tmp_inclusion]

    if len(inclusion_exclusion_split) == 2:
        inclusion, exclusion = inclusion_exclusion_split
    elif len(inclusion_exclusion_split) == 1:
        inclusion = inclusion_exclusion_split[0]
    else:
        return None

    inclusions = []
    if inclusion.strip():
        for criterion in re.split(r" - | \d\. ", inclusion):
            if criterion.strip() and criterion.strip() != ":":
                criterion = re.sub(r"[\r\n\t ]+", " ", criterion)
                inclusions.append(criterion)
    else:
        return None

    exclusions = []
    if exclusion.strip():
        for criterion in re.split(r" - | \d\. ", exclusion):
            if criterion.strip() and criterion.strip() != ":":
                criterion = re.sub(r"[\r\n\t ]+", " ", criterion)
                exclusions.append(criterion)

    return inclusions, exclusions


def parse_clinical_trials_from_folder(
    folder_name: str, first_n: Union[None, int] = None
) -> List[ClinicalTrial]:
    files = [y for x in os.walk(folder_name) for y in glob(os.path.join(x[0], "*.xml"))]

    if first_n:
        files = files[:first_n]

    total_parsed = 0

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
                result = parse_criteria(criteria=criteria)
                if result:
                    total_parsed += 1

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

    print(f"percentage of successfully parsed criteria : {total_parsed / len(files)}")

    return clinical_trials
