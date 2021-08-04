from trec_cds.data.parsers import (
    parse_clinical_trials_from_folder,
    parse_topics_from_xml,
)

if __name__ == "__main__":
    topic_file = "../data/external/topics2021.xml"
    folder_name = "../data/external/ClinicalTrials"

    topics = parse_topics_from_xml(topic_file)

    cts = parse_clinical_trials_from_folder(folder_name=folder_name)
