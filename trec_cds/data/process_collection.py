import zipfile
import pandas as pd
import json
from tqdm import tqdm
from typing import List
import re
import numpy as np
from os.path import exists


def collection2jsonl(
        path: str = "/content/drive/MyDrive/trec_clinical_trials/data/raw/",
        out_file_name: str = "clinical_trials_2021-04-27",
        out_path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/"
) -> pd.DataFrame:
    """
    collection2jsonl reads the collection of trials from the zipped parts
    and writes a jsonl file with a dictionary per trial
    {docno: trail id, raw_document: raw xml as a string}.
    If already exists, skips
    """

    out_file = f"{out_path}/{out_file_name}.jsonl"
    if not exists(out_file):
        with open(out_file, "w") as f_out:
            part = "ClinicalTrials.2021-04-27.part{0}.zip"
            for i in tqdm(range(1, 6), desc="part"):
                with zipfile.ZipFile(f"{path}/{part.format(i)}", "r") as f:
                    for name in tqdm(f.namelist(), desc="file"):
                        if name[-3:] == "xml":
                            entry = {
                                "docno": name[-15:-4],
                                "raw_document": f.read(name).decode(encoding='UTF-8')
                            }
                            f_out.write(
                                json.dumps(entry)
                            )
                            f_out.write("\n")


def search_fields(
        field: str,
        x: str
):
    rex_fields = "^.*?<{0}>(?:.*?<textblock>)?(.*?)(?:</textblock>.*?)?</{0}>.*?$"
    try:
        return re.search(rex_fields.format(field), x, re.DOTALL).groups(0)[0]
    except:
        return np.nan


def split_collection(
        collection: str = "clinical_trials_2021-04-27.jsonl",
        out_path: str = "/content/drive/MyDrive/trec_clinical_trials/data/interim/",
        out_file_name: str = "split_clinical_trials_2021-04-27.jsonl",
        cols_4_index: List = [
            "official_title",
            "brief_title",
            "condition",
            "brief_summary",
            "detailed_description",
            "criteria"
        ]
):
    """
    split_collection reads the jsonl collection with trials as
    {docno: trail id, raw_document: raw xml as a string}
    and writes another jsonl as
    {docno: trail id, [cols_4_index]:[split content of the trial based on fields]}.
    If already exists, skips
    """

    out_file = f"{out_path}/{out_file_name}"
    if not exists(out_file):
        n = 1000  # chunk row size

        fields = [
            "gender",
            "minimum_age",
            "maximum_age",
            "healthy_volunteers",
        ]

        with open(f"{out_path}/{collection}", "r") as f:
            with open(out_file, "w") as f_out:
                chunk = []
                for line in tqdm(f, desc="document"):
                    chunk.append(json.loads(line))

                    if len(chunk) == n:

                        chunk = pd.DataFrame(chunk)

                        for field_i in tqdm(cols_4_index + fields, desc="field"):
                            chunk[field_i] = chunk.raw_document.apply(
                                lambda x: search_fields(field_i, x)
                            )

                        chunk.drop(columns=["raw_document"], inplace=True)

                        chunk = chunk.to_dict('records')

                        for i in tqdm(chunk, desc="saving"):
                            f_out.write(json.dumps(i))
                            f_out.write("\n")

                        chunk = []