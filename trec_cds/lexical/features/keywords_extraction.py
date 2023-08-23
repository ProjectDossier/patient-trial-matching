from typing import Union, List

from gensim.parsing import remove_stopwords
from keybert import KeyBERT


class KeywordExtraction:
    def __init__(self):
        kw_model = KeyBERT(model="all-mpnet-base-v2")
        self.kw_model = kw_model

    def extract_keywords(self, text: Union[str, List[str]], verbose: bool = False):
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
            diversity=0.5,
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
            topics["keywords"] = self.extract_keywords(list(topics["query"]))

        elif mode == "dynamic_n_keys":
            topics["keywords"] = topics["query"].apply(
                lambda x: self.extract_keywords(x)
            )
        return topics
