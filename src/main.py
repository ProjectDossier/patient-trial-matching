import xml.etree.ElementTree as ET

from src.data.topic import Topic


def parse_topics_from_xml(topic_file):
    tree = ET.parse(topic_file)
    root = tree.getroot()

    topics = []
    for elem in root:
        topics.append(Topic(elem.attrib["number"], elem.text))

    return topics


if __name__ == "__main__":
    topic_file = "../data/external/topics2021.xml"

    topics = parse_topics_from_xml(topic_file)

    print(topics)
