import numpy as np
import pandas as pd
import random
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
        self.splits = splits
        self.mode = mode

        random.seed(r_seed)

        if mode == "train":
            self.data_spliting()

    def data_spliting(self):
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

        if self.mode != "predict_no_labels":
            # TODO split data by topics
            qids = data.qid.unique()
            random.shuffle(qids)

            n_train = int(len(qids) * self.splits["train"])
            n_val = int(len(qids) * self.splits["val"])
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
            self.data_train = data_train[["qid", "docno"]].values.tolist()

            data_val = data[data.qid.isin(qids_val)].copy()
            self.data_val = data_val[["qid", "docno"]].values.tolist()

            data_test = data[data.qid.isin(qids_test)].copy()
            self.data_test = data_test[["qid", "docno"]].values.tolist()

    def tokenize_samples(self, texts):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            model_max_length=512
        )

        tokenized_text = tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation='only_second',
            return_tensors="pt",
            add_special_tokens=True,
            return_token_type_ids=True,
        )

        return tokenized_text

    def build_train_batch(self, sample_ids: List):

        queries = self.db.get_queries(sample_ids)

        p_examples, n_examples = [], []

        # TODO get text for positive examples and add hard negatives from bm25 runs
        #   why not to use only samples from qrels
        #  - to have control over the complete experiment

        p_examples = p_examples[:self.batch_size // 2]
        n_examples = n_examples[:self.batch_size // 2]

        batch = p_examples + n_examples

        batch, labels, sample_ids = self.build_pred_batch(batch)

        return batch, labels, sample_ids

    def build_eval_batch(self, sample_ids: List):
        batch = list(self.test.loc[sample_ids].text)
        batch = self.tokenize_samples(batch)
        try:
            labels = list(self.test.loc[sample_ids].label)
        except AttributeError:
            labels = [-1] * len(sample_ids)
        return batch, labels, sample_ids


if __name__== "__main__":
    BatchProcessing()
