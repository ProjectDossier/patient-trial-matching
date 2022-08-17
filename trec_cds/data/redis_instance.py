import json
from tqdm import tqdm
from redis import StrictRedis
from typing import Union, Optional, List


class RedisInstance:
    def __init__(
            self,
            id: Union[str, int] = 0,
            path_to_collection: Optional[str] = None
    ):

        self.redis_db = StrictRedis(
            host="localhost",
            port=6379,
            db=id,
            charset="utf-8",
            decode_responses=True
        )

        try:
            self.get_docs(["NCT00000107"])
        except AssertionError:
            try:
                self.load_docs(path_to_collection)
            except TypeError:
                print("Warning: empty database")
                pass

    def delete_intances(self):
        self.redis_db.flushall()

    def load_docs(
            self,
            path: str
    ):
        fields = [
            "official_title",
            "brief_summary",
            "detailed_description",
            "condition",
            "criteria"
        ]

        with open(path) as f:
            for line in tqdm(f):
                doc = json.loads(line)
                docno = doc["docno"]

                insert = {}
                [
                    insert.update(
                        {
                            f"doc:{docno}:{field}": doc[field]
                        }
                    )
                    for field in fields
                ]

                self.redis_db.mset(insert)

    def get_docs(
            self,
            docnos: List[str],
            fields: List[str] = [
                "official_title",
                "brief_summary",
                "detailed_description",
                "condition",
                "criteria"
            ]
    ):
        n_fields = len(fields)

        keys = [
            f"doc:{docno}:{field}"
            for docno in docnos
            for field in fields
        ]

        data = self.redis_db.mget(keys)

        data = [data[i: i + n_fields] for i in range(0, len(data), n_fields)]

        assert len([i for i in data[0] if i is None]) == 0, "some id does not exists in db"

        return [
            {**{"id": doc_id}, **dict(zip(fields, values))}
            for doc_id, values in zip(docnos, data)
        ]


if __name__ == "__main__":
    data_path = "../../data/interim/"
    input_file = "sample_split_clinical_trials_2021-04-27.jsonl"
    db = RedisInstance(path_to_collection=f"{data_path}/{input_file}")
    docnos = ['NCT00000102', 'NCT00000104', 'NCT00000105', 'NCT00000106']
    docs = db.get_docs(docnos)
    print()
