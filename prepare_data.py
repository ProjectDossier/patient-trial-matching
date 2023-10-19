import os

if __name__ == "__main__":
    qrels2021 = "https://trec.nist.gov/data/trials/qrels2021.txt"
    qrels2022 = "https://trec.nist.gov/data/trials/qrels2022.txt"

    os.system(f"wget {qrels2021} -nc -P data/external/")
    os.system(f"wget {qrels2022} -nc -P data/external/")

    os.system(
        "wget https://owncloud.tuwien.ac.at/index.php/s/xtvAkFbtDEvcB6Y/download -O models/NER_model.zip"
    )
    os.system("unzip -n models/NER_model.zip -d models/")
    os.system("rm models/NER_model.zip")

    # download sample submission
    os.system(
        "wget https://owncloud.tuwien.ac.at/index.php/s/dFLy0DkHWhfJWCP/download -nc -O data/submissions/bm25_postprocessed_2021"
    )
    os.system(
        "wget https://owncloud.tuwien.ac.at/index.php/s/kt4M8t7Uag0O2Id/download -nc -O data/submissions/bm25_postprocessed_2022"
    )
