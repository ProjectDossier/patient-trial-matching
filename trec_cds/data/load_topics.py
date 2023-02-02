import re
from os.path import exists

import pandas as pd

from trec_cds.features.keywords_extraction import KeywordExtraction


def load_topics(
    path: str = "/content/drive/MyDrive/trec_clinical_trials/data/raw/",
    file_name: str = "topics2021",
    out_path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/",
    add_keywords_flag: bool = True,
):
    """
    load_topics takes the given topics and writes a csv table
    if add_keywords_flag = True, it adds keybert "keywords" column.
    If already exists, loads from file
    """
    out_file = f"{out_path}/{file_name}.csv"
    if not exists(out_file):
        with open(f"{path}/{file_name}.xml", "r") as f:
            topics = f.read()
            topic_re = '<topic number="(.*?)">(.*?)</topic>'
            topics = re.findall(topic_re, topics, re.DOTALL)
            topics = pd.DataFrame(
                [(i[0], i[1]) for i in topics], columns=["qid", "query"]
            )

            """some special characters not supported by pt"""
            topics["query"] = topics["query"].replace(
                "\\/|\\n|\*|\[|\]|'|\?|:", "", regex=True
            )

        if add_keywords_flag:
            add_keywords = KeywordExtraction().add_keywords
            topics = add_keywords(
                topics,
            )

        topics.to_csv(out_file, index=False)

    else:
        converters = {"query": str}
        topics = pd.read_csv(out_file, converters=converters)
        if add_keywords_flag and "keywords" not in topics.columns:
            add_keywords = KeywordExtraction().add_keywords
            topics = add_keywords(topics)

        # pyterrier works with str ids
        topics.qid = topics.qid.astype(str)
    return topics
