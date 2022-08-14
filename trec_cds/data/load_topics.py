from keybert import KeyBERT
from os.path import exists
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords
from typing import Union, List


class KeywordExtraction():
    def __init__(self):
        kw_model = KeyBERT(
            model="all-mpnet-base-v2"
        )
        self.kw_model = kw_model

    def extract_keywords(
            self,
            text: Union[str, List[str]],
            verbose: bool = False
    ):
        if text.__class__ == str:
            text = remove_stopwords(text)
            top_n = len(text.strip().split()) // 2
        else:
            top_n = 35

        keywords = self.kw_model.extract_keywords(
            text,
            top_n=top_n,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            diversity=0.5
        )

        if text.__class__ == str:
            print(f"adding {len(keywords)} keywords")
            if verbose:
                print(f"from:\n {text}\n to:\n {keywords}")
            return " ".join([i[0] for i in keywords])
        else:
            keys = []

            for i in keywords:
                keys.append([j[0] for j in i])
            return [" ".join(i) for i in keys]

    def add_keywords(self, topics, mode: str = "dynamic_n_keys"):
        if mode == "constant_n_keys":
            topics["keywords"] = self.extract_keywords(
                list(topics["query"])
            )

        elif mode == "dynamic_n_keys":
            topics["keywords"] = topics["query"].apply(
                lambda x: self.extract_keywords(x)
            )
        return topics


def load_topics(
        path: str = "/content/drive/MyDrive/trec_clinical_trials/data/raw/",
        file_name: str = "topics2021",
        out_path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/",
        add_keywords_flag: bool = True
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
                [(i[0], i[1]) for i in topics],
                columns=["qid", "query"]
            )

            """some special characters not supported by pt"""
            topics['query'] = topics['query'].replace(
                '\\/|\\n|\*|\[|\]|\'|\?|:', '', regex=True
            )

        if add_keywords_flag:
            add_keywords = KeywordExtraction().add_keywords
            topics = add_keywords(topics, )

        topics.to_csv(out_file, index=False)

    else:
        converters = {"query": str}
        topics = pd.read_csv(
            out_file,
            converters=converters
        )
        if add_keywords_flag and "keywords" not in topics.columns:
            add_keywords = KeywordExtraction().add_keywords
            topics = add_keywords(topics)

        # pyterrier works with str ids
        topics.qid = topics.qid.astype(str)
    return topics
