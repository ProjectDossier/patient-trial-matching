import json
import numpy as np
import pandas as pd
import random

from os.path import join as join_path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight as c_weights
from torch import tensor, LongTensor, FloatTensor
from transformers import AutoTokenizer
from typing import Dict, List, Optional


class BatchProcessing:
    def __init__(
        self,
        path: str = "../../data/raw/",
        mode: str = "train",
        splits: Dict = {"train": 0.60, "val": 0.10, "test": 0.30},
        r_seed: int = 42,
        tokenizer_name: str = "bert-base-uncased",
        train_batch_size: int = 16,
        n_val_samples: Optional[int] = None,
        n_test_samples: Optional[int] = None,
    ):
        self.train_batch_size = train_batch_size
        self.tokenizer_name = tokenizer_name

        random.seed(r_seed)

        if mode == "train":
            # todo connect redis server
            # TODO build datasets
            #  read bm25 run or take it from db

            data = pd.read_csv(
                "../../reports/DoSSIER_1.txt",
                header=None,
                names=[
                    "qid",
                    "Q0",
                    "docno",
                    "rank",
                    "score",
                    "run_id"
                ]
            )

            self.data = data

            # TODO split data by topics
            qids = data.qid.unique()
            random.shuffle(qids)

            n_train = int(len(qids) * splits["train"])
            n_val = int(len(qids) * splits["val"])
            n_test = len(qids) - (n_train + n_val)

            qids_train = qids[:n_train]
            qids_val = qids[n_train: n_train + n_val]
            qids_test = qids[-n_test:]

            # TODO how to present the input to the datamodule
            #  idea: pick all possible positive pairs from the runs
            #  and then get hard negatives in the batch processing

            # TODO get qrels here
            qrels = pd.read_csv(
                "../../data/raw/qrels_Judgment-of-0-is-non-relevant-1-is-excluded-and-2-is-eligible.txt",
                header=None,
                names=[
                    "qid",
                    "Q0",
                    "docno",
                    "label",
                ],
                sep=" "
            )
            data = data.merge(
                qrels,
                on=[
                    "qid",
                    "docno"
                ],
                how="left"
            )

            data = data.fillna(0)

            data_train = data[data.qid.isin(qids_train)].copy()

            # TODO positive examples are 1 and 2 labels for descriptive fields
            data_train = data_train[data_train.label.isin([1, 2])]
            data_train = data_train[["qid", "docno"]].values.tolist()

            data_val = data[data.qid.isin(qids_val)].copy()
            data_val = data_val[["qid", "docno"]].values.tolist()

            data_test = data[data.qid.isin(qids_test)].copy()
            data_test = data_test[["qid", "docno"]].values.tolist()


