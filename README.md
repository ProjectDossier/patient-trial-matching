patient-trial-matching
==============================

This package serves as basis for the paper "Effective Matching of Patients to Clinical Trials using Entity Extraction and Neural Re-ranking"



[![arXiv](https://img.shields.io/badge/arXiv-2307.00381-b31b1b.svg)](https://arxiv.org/abs/2307.00381)


## Installation

Tested with `python=3.8`. Assuming you have conda installed, create new environment with:

```bash
conda create --name patient-trial-matching python=3.8
```

Activate the environment:

```bash
conda activate patient-trial-matching
```

Install requirements:

```bash
$ pip install -r requirements.txt
$ git submodule update --init --recursive
$ pip install -e clinical-trials 
$ pip install spacy==3.1.6   # this will raise a warning, but it is needed for the models to properly load
$ pip install medspacy==0.2.0.0
$ pip install medspacy==0.2.0.1 
$ pip install pydantic==1.10.11 
```

This will install all required packages and also this project in a devel mode.


[Install](https://redis.io/docs/getting-started/installation/) and launch redis server:
```bash
redis-server
```
If your system does not support redis, the code will use the mockup version of the redis server.


## Data

Patients and clinical trials information can be downloaded from [TREC-CDS](http://trec-cds.org/2022.html) website:

* `topics2021.xml` file with 75 patients' data
* `topics2022.xml` file with 50 patients' data
* 5 .zip files with ClinicalTrials data


To download qrels and NER model for detecting age and gender run:

```bash
$ python prepare_data.py
```

ClinicalTrials should be extracted into `data/external/ClinicalTrials/` folder.


## Usage

You can test installation by running:

```bash
$ python trec_cds/main.py
```

### Features

- Training and inference of custom entity recognition model based on spacy NER
- Indexing with BM25
- Postprocessing using topic and clinical trial related features
- re-ranking using neural model and eligibility criteria


## Citing

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
