patient-trial-matching
==============================

This package serves as basis for the paper "Effective Matching of Patients to Clinical Trials using Entity Extraction
and Neural Re-ranking"

[![arXiv](https://img.shields.io/badge/arXiv-2307.00381-b31b1b.svg)](https://arxiv.org/abs/2307.00381)

### Table of contents

1. [Installation](#installation)
2. [Data](#data)
3. [Usage](#usage)
4. [Citing](#citing)

## <a name='installation' />  1. Installation

Tested with `python=3.8`. Assuming you have conda installed, create new environment with:

```bash
$ conda create --name patient-trial-matching python=3.8
```

Activate the environment:

```bash
$ conda activate patient-trial-matching
```

Install requirements:

```bash
(patient-trial-matching)$ pip install -r requirements.txt
(patient-trial-matching)$ git submodule update --init --recursive
(patient-trial-matching)$ pip install -e clinical-trials 
(patient-trial-matching)$ pip install spacy==3.1.6   # this will raise a warning, but it is needed for the models to properly load
(patient-trial-matching)$ pip install medspacy==0.2.0.0
(patient-trial-matching)$ pip install medspacy==0.2.0.1 
(patient-trial-matching)$ pip install pydantic==1.10.11 
```

This will install all required packages and also this project in a devel mode.

[Install](https://redis.io/docs/getting-started/installation/) and launch redis server:

```bash
redis-server
```

If your system does not support redis, the code will use the mockup version of the redis server.

## <a name='data' /> 2. Data

Patients and clinical trials information can be downloaded from [TREC-CDS](http://trec-cds.org/2022.html) website:

* `topics2021.xml` file with 75 patients' data
* `topics2022.xml` file with 50 patients' data
* 5 .zip files with ClinicalTrials data

ClinicalTrials XMLs should be extracted into `data/external/ClinicalTrials/` folder.

To download qrels and NER model for detecting age and gender run:

```bash
(patient-trial-matching)$ python prepare_data.py
```

## <a name='usage' /> 3. Usage

### 3.1 Data preprocessing

Prepare patients' data. It will prepare 2 files: `topics2021.jsonl` and `topics2022.jsonl` in output_folder folder.

```bash
(patient-trial-matching)$ python trec_cds/data/convert_trials_to_jsonl.py --input_folder PATH_TO_PATIENT_XML_DATA --output_folder PATH_TO_OUTPUT_FOLDER
```

Prepare trials' data. This might take several hours (5-10 hours) as the entity extraction model is making predictions
for each trial.

```bash
(patient-trial-matching)$ python trec_cds/data/convert_trials_to_jsonl.py --input_data PATH_TO_UNZIPPED_XML_DATA --outfile PATH_TO_OUTPUT_FOLDER
```

Both these scripts will parse data and extract drug and disease entities. Processed output will be generated in a jsonl
format.

### 3.2 Lexical matching

The experiments for input fields and extracted keywords below are for the BM25 and BM25+ models. To run other models (
DFR, TF-IDF), you need to install pyTerrier and run the corresponding terrier scripts.

#### 3.2.1 Input fields experiment

To run the lexical matching experiment, run:

```bash
(patient-trial-matching)$ python scripts/input_fields_experiment.py --topic_file data/external/topics2021.xml --clinical_trials_folder PATH_TO_UNPROCESSED_CLINICAL_TRIALS  --binary_qrels data/external/qrels2021_binary.txt --graded_qrels data/external/qrels2021.txt --results_folder RESULTS_OUTPUT_FOLDER  --submission_folder TREC_SUBMISSION_FOLDER
```

#### 3.2.2 Extracted keywords experiment

To run the experiment measuring impact of extracted keywords, run:

```bash
(patient-trial-matching)$ python scripts/extracted_keywords_experiment.py --trials_file PATH_TO_TRIALS_JSONL_FILE --topic_file data/external/topics2021.jsonl  --binary_qrels data/external/qrels2021_binary.txt --graded_qrels data/external/qrels2021.txt --results_folder RESULTS_OUTPUT_FOLDER  --submission_folder TREC_SUBMISSION_FOLDER 
```

### 3.3 Postprocessing / filtering

Postprocessing script is based on the detected gender and age of patients. It will filter out trials that are not
suitable for a given patient.

```bash
(patient-trial-matching)$ python scripts/filtering_experiment.py --topic_file data/external/topics2021.jsonl --binary_qrels data/external/qrels2021_binary.txt --graded_qrels data/external/qrels2021.txt --runs_folder PATH_TO_FOLDER_WITH_PREVIOUS_STEP_OUTPUTS --output_folder RESULTS_OUTPUT_FOLDER 
```

### 3.4 Neural reranking

#### 3.4.1 Topical training

All parameters and configurations are stored in yml files inside `config` folder.
Before running the models ensure that these configs are pointing to correct data paths.

To train the neural model, run:

```bash
(patient-trial-matching)$ python neural/models/train_crossencoder.py
```

#### 3.4.2 Criteria training

To train the neural model, run:

```bash
(patient-trial-matching)$ python neural/models/further_train_crossencoder.py
```

#### 3.4.3 Inference

To run inference, run:

```bash
(patient-trial-matching)$ python neural/models/predict_crossencoder.py
```

## <a name='citing' /> 4. Citing

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
