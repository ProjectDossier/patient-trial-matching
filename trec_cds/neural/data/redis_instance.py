import json
from typing import Union, Optional, List

import pandas as pd
from numpy import nan
from redis import StrictRedis
from tqdm import tqdm

from trec_cds.data.load_data_from_file import load_jsonl


class RedisInstance:
    def __init__(
        self,
        id: Union[str, int] = 1,
        path_to_collection: Optional[str] = None,
        path_to_topics: Optional[str] = None,
        version: Optional[str] = "2021",
    ):

        self.redis_db = StrictRedis(
            host="149.132.176.45",
            port=6379,
            db=id,
            charset="utf-8",
            decode_responses=True,
            password="academic-coll",
            socket_keepalive=True,
            socket_timeout=259200,
            socket_connect_timeout=259200,
        )

        self.str_fields = [
            "org_study_id",
            "brief_title",
            "official_title",
            "brief_summary",
            "detailed_description",
            "study_type",
            "criteria",
            "gender",
        ]

        self.list_fields = [
            "inclusion",
            "exclusion",
            "primary_outcomes",
            "secondary_outcomes",
            "conditions",
        ]

        self.dict_fields = ["interventions"]

        self.bool_fields = ["accepts_healthy_volunteers"]

        self.float_fields = ["minimum_age", "maximum_age"]

        try:
            self.get_docs(["NCT00000107"])
            self.get_topics([1], version)
        except AssertionError:
            if path_to_collection is not None:
                self.load_docs(path_to_collection)
            else:
                print("Warning: empty collection")
            if path_to_topics is not None:
                self.load_topics(path_to_topics, version)
            else:
                print("Warning: empty topics")

    def delete_intances(self):
        self.redis_db.flushall()

    def load_docs(self, path: str = "../../data/interim/trials_parsed.jsonl"):
        fields = []
        with open(path) as f:
            for line in tqdm(f):

                doc = json.loads(line)
                docno = doc["nct_id"]

                if len(fields) == 0:
                    fields = list(set(doc.keys()) - set(docno))

                insert = {}

                for field in fields:
                    if doc[field] in [None, nan, "nan"]:
                        continue
                    elif field in self.str_fields:
                        if len(doc[field]) == 0:
                            continue
                    elif field in self.list_fields:
                        if len(doc[field]) == 0:
                            continue
                        doc[field] = "|".join(doc[field])
                    elif field in self.dict_fields:
                        if len(doc[field]) == 0:
                            continue
                        doc[field] = "|".join([json.dumps(i) for i in doc[field]])
                    elif field in self.bool_fields:
                        doc[field] = str(doc[field])

                    insert.update({f"doc:{docno}:{field}": doc[field]})

                self.redis_db.mset(insert)

    def get_docs(self, docnos: List[str], fields: List[str] = None):
        if fields is None:
            fields = [
                "nct_id",
                "brief_title",
                "official_title",
                "brief_summary",
                "detailed_description",
                "study_type",
                "criteria",
                "gender",
                "inclusion",
                "exclusion",
                "conditions",
                "interventions",
                "accepts_healthy_volunteers",
                "minimum_age",
                "maximum_age",
            ]
        n_fields = len(fields)

        keys = [f"doc:{docno}:{field}" for docno in docnos for field in fields]

        data = self.redis_db.mget(keys)

        data = [data[i : i + n_fields] for i in range(0, len(data), n_fields)]

        assert (
            len([i[0] for i in data if all(i is None for j in i)]) == 0
        ), "some id does not exists in db"

        result = []
        for values in data:
            item = {}
            for field, value in zip(fields, values):
                if field in self.list_fields:
                    if value is None:
                        value = []
                    else:
                        value = value.split("|")
                elif field in self.dict_fields:
                    if value is None:
                        value = []
                    else:
                        value = [json.loads(i) for i in value.split("|")]
                elif field in self.bool_fields:
                    if value is not None:
                        value = bool(value)
                elif field in self.float_fields:
                    if value is not None:
                        value = float(value)

                item.update({field: value})
            result.append(item)

        return result

    def load_topics(self, path: str, version: str, fields: List[str] = None):
        if fields is None:
            fields = [
                "qid",
                "query",
                "keywords",
                "description",
                "gender",
                "age",
            ]
        topics = pd.read_csv(path)

        for idx, topic in tqdm(topics.iterrows()):
            qid = topic["qid"]

            insert = {}

            for field in fields:
                insert.update({f"topic:{version}{qid}:{field}": str(topic[field])})

            self.redis_db.mset(insert)

    def get_topics(self, qids: List[int], version: str, fields: List[str] = None):

        if fields is None:
            fields = [
                "qid",
                "query",
                "description",
                "keywords",
                "gender",
                "age",
            ]
        n_fields = len(fields)

        keys = [f"topic:{version}{qid}:{field}" for qid in qids for field in fields]

        data = self.redis_db.mget(keys)

        data = [data[i : i + n_fields] for i in range(0, len(data), n_fields)]

        assert (
            len([i[0] for i in data if i[0] is None]) == 0
        ), "some id does not exists in db"

        result = []
        for values in data:
            item = {}
            for field, value in zip(fields, values):
                if field == "age" and value is not None:
                    value = float(value)
                item.update({field: value})
            result.append(item)
        return result

    def filter_run(self, qid: List[int], docno: List[int]):
        patient = self.get_topics(qids=[qid], fields=["age", "gender"])[0]
        trial = self.get_docs(
            docnos=[docno], fields=["gender", "minimum_age", "maximum_age"]
        )[0]

        trial["gender"] = "A" if trial["gender"] is None else trial["gender"]
        trial["minimum_age"] = (
            -1 if trial["minimum_age"] is None else trial["minimum_age"]
        )
        trial["maximum_age"] = (
            100 if trial["maximum_age"] is None else trial["maximum_age"]
        )

        result = (
            patient["gender"] == trial["gender"] or trial["gender"] not in ["F", "M"]
        ) and trial["minimum_age"] <= patient["age"] <= trial["maximum_age"]

        return result


class MockupInstance:
    """Mockup instance for situations where Redis is not available."""

    def __init__(
        self,
        parsed_trials_jsonl: Optional[str] = None,
        path_to_topics: Optional[str] = None,
        version: Optional[int] = 2021,
    ):
        if parsed_trials_jsonl is None:
            trials_file = "../../data/processed/trials_parsed.jsonl"
        else:
            trials_file = parsed_trials_jsonl
        trials = load_jsonl(trials_file)

        self.cts_dict = {ct["nct_id"]: ct for ct in trials}

        self.patients = []
        if path_to_topics is None:
            path_to_topics = "../../data/processed/topics2021.jsonl"
        patients = load_jsonl(path_to_topics)
        self.patients.extend(patients)
        # for patient_file in ['topics2021', 'topics2022']:
        # for patient_file in ["topics2021"]:
        #     infile = f"/home/wkusa/projects/TREC/trec-cds1/data/processed/{patient_file}.jsonl"
        #     patients = load_jsonl(infile)
        #     self.patients.extend(patients)

        self.patients_dicts = {}
        self.patients_dicts[version] = {str(p["patient_id"]): p for p in self.patients}

    def get_topics(
        self, qids: List[int], fields: List[str] = None, version: int = 2021
    ):
        patients_dict = self.patients_dicts[version]

        if fields is None:
            fields = ["qid", "query", "keywords", "gender", "age"]
        result = []
        for qid in qids:
            patient = patients_dict[qid]

            item = {field: patient[field] for field in fields}
            result.append(item)

        return result

    def get_docs(self, docnos: List[str], fields: List[str] = None):
        if fields is None:
            fields = [
                "nct_id",
                "brief_title",
                "official_title",
                "brief_summary",
                "detailed_description",
                "study_type",
                "criteria",
                "gender",
                "inclusion",
                "exclusion",
                "conditions",
                "interventions",
                "accepts_healthy_volunteers",
                "minimum_age",
                "maximum_age",
            ]
        result = []
        for docno in docnos:
            ct = self.cts_dict[docno]

            item = {}
            for field in fields:
                item.update({field: ct[field]})
            result.append(item)

        return result


if __name__ == "__main__":
    version = "2021"

    db_instance = RedisInstance(version=version)
