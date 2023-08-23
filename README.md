patient-trial-matching
==============================

This package serves as basis for the paper "Effective Matching of Patients to Clinical Trials using Entity Extraction and Neural Re-ranking"

[![arXiv](https://img.shields.io/badge/arXiv-2307.00381-b31b1b.svg)](https://arxiv.org/abs/2307.00381)

### Table of contents
1. [Installation](#installation)
3. [Data](#data)
2. [Usage](#usage)


## <a name='installation' />  1. Installation

Tested with `python=3.8`. Install requirements with:

```bash
conda create --name patient-trial-matching python=3.8
```

```bash
conda activate patient-trial-matching
```

```bash
(patient-trial-matching)$ pip install -r requirements.txt
(patient-trial-matching)$ git submodule update --init --recursive
(patient-trial-matching)$ pip install -e clinical-trials 
```

This will install all required packages and also this project in a devel mode.


[Install](https://redis.io/docs/getting-started/installation/) and launch redis server:
```bash
redis-server
```


## <a name='data' /> 2. Data

Patients and clinical trials information can be downloaded from [TREC-CDS](http://trec-cds.org/2022.html) website:

* `topics2021.xml` file with 75 patients' data
* `topics2022.xml` file with 50 patients' data
* 5 .zip files with ClinicalTrials data

ClinicalTrials should be extracted into `data/external/ClinicalTrials/` folder.


## <a name='usage' /> 3. Usage

You can test installation by running:

```bash
(patient-trial-matching)$ python trec_cds/main.py
```

### 3.1 Data preparation

Prepare patient's data. It will prepare 2 files: `topics2021.jsonl` and `topics2022.jsonl` in output_folder folder.

```bash
(patient-trial-matching)$ python trec_cds/data/convert_trials_to_jsonl.py --input_folder PATH_TO_PATIENT_XML_DATA --output_folder PATH_TO_OUTPUT_FOLDER
```

Prepare trials' data. This might take several hours (5-10 hours) as the entity extraction model is making predictions for each trial.

```bash
(patient-trial-matching)$ python trec_cds/data/convert_trials_to_jsonl.py --input_data PATH_TO_UNZIPPED_XML_DATA --outfile P
```

### 3.2 Lexical matching







### Features

- Training and inference of custom entity recognition model based on spacy NER
- Indexing with BM25
- Postprocessing using topic and clinical trial related features
- re-ranking using neural model and eligibility criteria


## 4. Citing

If you find our code useful, please cite our paper:

```bibtex
@article{Kusa2023Effective,
title = {Effective matching of patients to clinical trials using entity extraction and neural re-ranking},
journal = {Journal of Biomedical Informatics},
pages = {104444},
year = {2023},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2023.104444},
url = {https://www.sciencedirect.com/science/article/pii/S153204642300165X},
author = {Wojciech Kusa and Óscar E. Mendoza and Petr Knoth and Gabriella Pasi and Allan Hanbury}
}
```
