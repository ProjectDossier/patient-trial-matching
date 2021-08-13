from rank_bm25 import BM25Okapi
from trec_cds.data.parsers import parse_clinical_trials_from_folder, parse_topics_from_xml
import spacy
import pickle
import tqdm
import numpy as np


corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)
# <rank_bm25.BM25Okapi at 0x1047881d0>


query = "windy London"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
# array([0.        , 0.93729472, 0.        ])
print(doc_scores)

if __name__ == '__main__':
    folder_name = "data/external/ClinicalTrials"
    first_n = 2000

    cts = parse_clinical_trials_from_folder(folder_name=folder_name, first_n=first_n)
    # sample

    nlp = spacy.load("en_core_web_sm")

    cts_tokenized = []
    for clinical_trial in tqdm.tqdm(cts):
        if clinical_trial.criteria is None or clinical_trial.criteria.strip() == "":
            doc = nlp('empty')
            print("empty")
            print(clinical_trial.summary)
        else:
            doc = nlp(clinical_trial.criteria)
        cts_tokenized.append([word.text for word in doc])

    print('tokenizing done')
    bm25 = BM25Okapi(cts_tokenized)
    print(bm25)
    pickle.dump(bm25, open("models/bm25.p", "wb"))

    topic_file = "data/external/topics2021.xml"
    topics = parse_topics_from_xml(topic_file)


    for topic in topics[:3]:
        doc = nlp(topic.text)
        doc_scores = bm25.get_scores([word.text for word in doc])
        print(doc_scores)

