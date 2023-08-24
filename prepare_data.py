import os

if __name__ == "__main__":
    qrels2021 = "https://trec.nist.gov/data/trials/qrels2021.txt"
    qrels2022 = "https://trec.nist.gov/data/trials/qrels2022.txt"

    os.system(f"wget {qrels2021} -P data/external/")
    os.system(f"wget {qrels2022} -P data/external/")

    topics2021 = "http://www.trec-cds.org/topics2021.xml"
    topics2022 = "http://www.trec-cds.org/topics2022.xml"

    os.system(f"wget {topics2021} -P data/external/")
    os.system(f"wget {topics2022} -P data/external/")

    os.system(
        "wget https://owncloud.tuwien.ac.at/index.php/s/xtvAkFbtDEvcB6Y/download -O models/NER_model.zip"
    )
    os.system("unzip models/NER_model.zip -d models/")
    os.system("rm models/NER_model.zip")
