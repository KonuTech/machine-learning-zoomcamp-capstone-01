# Used car price prediction

## Objective

This repository contains the midterm project for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course provided by [DataTalks.Club](https://datatalks.club/).

The goal of the project is to apply what we have learned during the course. This project aims to develop an exemplary machine learning application that predicts whether a customer will end up as a defaulted one.

## Dataset

The dataset used to feed the MLOps pipeline was downloaded from [Kaggle](https://www.kaggle.com/competitions/amex-default-prediction).

Reading data in chunks was applied to handle the size of the training dataset, which is approximately 16 GB, and the test set, which is around 32 GB.

The datasets were published by American Express as part of a prediction competition with a total prize pool of $100,000. 

My best submission achieved a Normalized Gini Coefficient score of 0.57895, which placed my late submission somewhere below the 4000th position out of more than 5000 in total.

### Applied technologies

| Name | Scope |
| --- | --- |
| Jupyter Notebooks | Exploratory data analysis and pipeline prototyping |
| Pandas | Feature engineering |
| Scikit-learn | Training pipeline, including Feature selection |
| XGBoost | Classifier |
| Flask | Web server |
| pylint | Python static code analysis |
| black | Python code formatting |
| isort | Python import sorting |

### Steps to run the scoring app

1. Clone the [machine-learning-zoomcamp-capstone-01](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-01.git) repository:

    ```bash
    $ git clone https://github.com/KonuTech/machine-learning-zoomcamp-capstone-01.git
    ```

2. Install the pre-requisites necessary to run scoring app:

    ```bash
    $ pip install requrements.txt
    ```
3. Move to the scoring directory
    ```bash
    $ cd scoring/
    ```
3. Build the docker image
    ```bash
    $ docker build -t machine-learning-zoomcamp-capstone-01 .
    ```
4. Run the docker image with expose
    ```bash
    $ docker run -it --rm -p 9696:9696 --entrypoint=bash machine-learning-zoomcamp-capstone-01
    ```
5. Once image is run, start Flask service
    ```bash
    $ python predict.py
    ```
5. From the local terminal run the script which scores two exemplary customers. One should be scored as Bad, the other as Good
    ```bash
    $ cd scoring/
    $ python predict-test.py
    ```

The output shoud looks as the one on below screen shot:

<img src="notebooks/exemplary_scoring_output.jpg" width="60%"/>



------------

Project Structure
------------
    |-- data/ (directory)
        |-- parquet_partitions/ (directory)
        |-- sample_submission.csv
        |-- submission_2023-11-01_17-54-58.csv
        |-- submission_2023-11-01_20-08-34.csv
        |-- submission_2023-11-04_18-06-41.csv
        |-- submission_2023-11-05_14-54-16.bin.csv
    |-- eda/ (directory)
        |-- histograms/ (directory)
    |-- models/ (directory)
        |-- grid_search_results_2023-11-01_17-54-58.json
        |-- grid_search_results_2023-11-01_20-08-34.json
        |-- grid_search_results_2023-11-02_16-48-32.json
        |-- grid_search_results_2023-11-04_18-06-41.json
        |-- grid_search_results_2023-11-05_14-54-16.json
        |-- grid_search_results_2023-11-05_22-12-26.json
        |-- LogisticRegression_2023-11-01_17-54-58.bin
        |-- LogisticRegression_2023-11-01_20-08-34.bin
        |-- LogisticRegression_2023-11-02_16-48-32.bin
        |-- LogisticRegression_2023-11-04_18-06-41.bin
        |-- LogisticRegression_2023-11-05_14-54-16.bin
        |-- LogisticRegression_2023-11-05_22-12-26.bin
        |-- XGBoost_2023-11-01_17-54-58.bin
        |-- XGBoost_2023-11-01_20-08-34.bin
        |-- XGBoost_2023-11-02_16-48-32.bin
        |-- XGBoost_2023-11-04_18-06-41.bin
        |-- XGBoost_2023-11-05_14-54-16.bin
        |-- XGBoost_2023-11-05_22-12-26.bin
    |-- notebooks/ (directory)
        |-- 01_prepare_train_data.ipynb
        |-- 02_eda.ipynb
        |-- 03_downsample_data.ipynb
        |-- 04_get_champion_binary_classifier.ipynb
        |-- 05_prepare_test_data.ipynb
        |-- 06_score_test_data.ipynb
        |-- dict_vectorizer.pkl
        |-- exemplary_scoring_output.jpg
    |-- scoring/ (directory)
        |-- imputer.pkl
        |-- training_log.log
        |-- dict_vectorizer.pkl
        |-- Dockerfile
        |-- example_bad_customer.json
        |-- example_good_customer.json
        |-- imputer.pkl
        |-- predict-test.py
        |-- predict.py
        |-- requirements.txt
        |-- XGBoost_2023-11-05_14-54-16.bin
    |-- scripts/ (directory)
        |-- correlation.py
        |-- project_tree.py
        |-- train.py
    |-- .git/ (directory)
    |-- .gitignore
    |-- Pipfile
    |-- Pipfile.lock
    |-- project_structure.txt
    |-- README.md
    |-- requirements.txt
    |-- requirements_dev.txt
    |-- temp.txt
    |-- training_log.log

### Peer review criterias - a self assassment:
* Problem description
    * 2 points: The problem is well described and it's clear what the problem the project solves
* Cloud
    * 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
* Experiment tracking and model registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 4 points: Fully deployed workflow 
* Model deployment
    * 2 points: Model is deployed but only locally
* Model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
* Reproducibility
    * 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.
* Best practices
    * There are unit tests (1 point)
    * Linter and/or code formatter are used (1 point)
    * There's a Makefile (1 point)
    * There are pre-commit hooks (1 point)
