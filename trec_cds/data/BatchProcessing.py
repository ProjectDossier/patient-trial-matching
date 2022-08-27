import pandas as pd
import random
from transformers import AutoTokenizer
from typing import Dict, List, Optional
from trec_cds.data.redis_instance import RedisInstance
import time


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
        # TODO truncate runs for debugging (evaluation)
        self.train_batch_size = train_batch_size
        self.tokenizer_name = tokenizer_name
        self.splits = splits
        self.mode = mode
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples

        random.seed(r_seed)

        time.sleep(60)
        self.db = RedisInstance()

        self.load_data()

    def load_data(self):

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

        if self.mode != "predict_w_no_labels":
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

            self.reference_run = data.copy()

            data = data.fillna(0)
            if self.mode == "train":
                qids = data.qid.unique()
                random.shuffle(qids)

                n_train = int(len(qids) * self.splits["train"])
                n_val = int(len(qids) * self.splits["val"])
                n_test = len(qids) - (n_train + n_val)

                qids_train = qids[:n_train]
                qids_val = qids[n_train: n_train + n_val]
                qids_test = qids[-n_test:]

                data_train = data[data.qid.isin(qids_train)].copy()

                # TODO positive examples are 1 and 2 labels for descriptive fields
                data_train = data_train[data_train.label.isin([1, 2])]
                self.data_train = data_train[["qid", "docno"]].values.tolist()

                data_val = data[data.qid.isin(qids_val)].copy()
                self.data_val = data_val[["qid", "docno"]].values.tolist()
                if self.n_val_samples is not None:
                    self.data_val = truncate_rank(qids_val, self.data_val, self.n_val_samples)

                data_test = data[data.qid.isin(qids_test)].copy()
                self.data_test = data_test[["qid", "docno"]].values.tolist()
                if self.n_test_samples is not None:
                    self.data_test = truncate_rank(qids_test, self.data_test, self.n_test_samples)

            self.data = data[["qid", "docno"]].values.tolist()

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

    def build_batch(self, sample_ids: List):
        qids = [i[0] for i in sample_ids]
        unique_qids = list(set(qids))
        topics = self.db.get_topics(
            unique_qids,
            ["query"]
        )

        topics_dict = {}
        for qid, query in zip(unique_qids, topics):
            topics_dict.update({qid: query})
        # TODO: parameterize fields?
        fields = [
            'conditions',
            'brief_title',
            'official_title',
            'brief_summary',
            'detailed_description',
        ]

        docnos = [i[1] for i in sample_ids]
        unique_docnos = list(set(docnos))
        docs = self.db.get_docs(
            unique_docnos,
            fields
        )

        docs_dict = {}
        for docno, doc in zip(unique_docnos, docs):
            docs_dict.update({docno: doc})

        sample_texts = []
        for qid, docno in sample_ids:
            query = topics_dict[qid]["query"]
            doc = docs_dict[docno]
            doc = " ".join(
                flatten_list(
                    [doc[i] for i in fields if doc[i] is not None]
                )
            )

            sample_texts.append(
                [
                    query,
                    doc
                ]
            )

        batch = self.tokenize_samples(sample_texts)
        return batch, qids, docnos

    def build_train_batch(self, sample_ids: List):

        # random sample from the batch pool
        random.shuffle(sample_ids)
        positives = sample_ids[: self.train_batch_size // 2]

        # picking hard negatives from reference run
        run = self.reference_run
        negatives = []
        for qid, docno in positives:
            negative_list = run[
                                (run.qid == qid) & (run.label == 0)
                                ].docno.values.tolist()[0:100]
            random.shuffle(negative_list)
            negatives.append([qid, negative_list[0]])

        sample_ids = positives + negatives

        batch, _, _ = self.build_batch(sample_ids)

        return batch


def flatten_list(lst):
    flst = []
    for i in lst:
        if isinstance(i, list):
            flst.extend(flatten_list(i))
        else:
            flst.append(i)
    return flst


def truncate_rank(qids, data, n_samples):
    data_val = []
    for qid in qids:
        for idx, pair in enumerate([pair for pair in data if pair[0] == qid]):
            if idx == n_samples:
                break
            data_val.append(pair)

    return data_val
