from features.ner import get_ner_model, extract_age, extract_gender, extract_gender_from_text
from trec_cds.data.parsers import parse_topics_from_xml
from trec_cds.data.utils import Gender

if __name__ == "__main__":
    topic_file = "data/external/topics2021.xml"
    folder_name = "data/external/ClinicalTrials"

    topics = parse_topics_from_xml(topic_file)
    print(topics)

    nlp = get_ner_model()

    for topic in topics:
        doc = nlp(topic.text)

        age_entities = [ent.text for ent in doc.ents if ent.label_ == "AGE"]
        if len(age_entities) > 0:
            topic.age = extract_age(age_entities[0])
        else:
            topic.age = -1

        gender_entities = [ent.text for ent in doc.ents if ent.label_ == "GENDER"]
        if len(gender_entities) > 0:
            topic.gender = extract_gender(gender_entities[0])
        else:
            topic.gender = extract_gender_from_text(topic.text)


    # cts = parse_clinical_trials_from_folder(folder_name=folder_name)
    # print(cts)
