from typing import List
from tqdm import tqdm
import pyterrier as pt
import json
from os.path import exists

if not pt.started():
    pt.init()


def collection_iter(
        path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/",
        collection: str = "split_clinical_trials_2021-04-27.jsonl",
        cols_4_index: List = [
            "official_title",
            "brief_title",
            "condition",
            "brief_summary",
            "detailed_description",
            "criteria"
        ]
):
    with open(f"{path}/{collection}", "r") as f:
        for line in tqdm(f, desc="indexing"):
            row = json.loads(line)
            document = '\n'.join([row[i] for i in cols_4_index if row[i].__class__ == str])
            yield {'docno': row['docno'], 'text': document}


def get_retriever(
        path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/iter_index",
        cols_4_index: List[str] = [
            "official_title",
            "brief_summary",
            "detailed_description",
            "condition",
            "criteria"
        ],
        config_top_k: int = 1000
):
    index = f"{path}/data.properties"

    if not exists(index):
        iter_indexer = pt.IterDictIndexer(path, blocks=True, overwrite=True)
        doc_iter = collection_iter(cols_4_index=cols_4_index)
        indexref = iter_indexer.index(doc_iter)
    else:
        indexref = pt.IndexFactory.of(index)

    retr_controls = {
        "wmodel": "BM25",
        "string.use_utf": "true",
        "end": config_top_k
    }

    retr = pt.BatchRetrieve(indexref, controls=retr_controls)

    return retr
