# RYANSQL
## Introduction
A source code for RYANSQL, a text-to-SQL system for complex, cross-domain databases.

Reference Paper: Choi et al., [RYANSQL: Recursively Applying Sketch-based Slot Fillings for Complex Text-to-SQL in Cross-Domain Databases](https://arxiv.org/abs/2004.03125), 2020

The system is submitted to [SPIDER leaderboard](https://yale-lily.github.io/spider). The system and its minor improved version RYANSQL v2 is ranked at second and fourth place (as of February 2020).

The system does NOT use any database records, which make it more acceptable to the real world company applications. Among the systems not using the database records, the system is ranked #1 in the leaderboard.

## Requirements
Python3 <br>
Tensorflow 1.14 <br>
nltk

## Install 
Download the [BERT](https://github.com/google-research/bert) pretrained model. You can only download the model, not the whole git. The system uses [BERT-large, uncased with Whole Word Masking model](https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip). Unzip the downloaded file.
 
Download the SPIDER dataset from https://yale-lily.github.io/spider. Unzip the downloaded file.

## Train 
Run:

```
python src/trainer.py [BERT_DIR] [SPIDER_DATASET_DIR]
```

An example is:

```
python src/trainer.py ./wwm_uncased_L-24_H-1024_A-16 ./spider
```

The training takes about a day or 2 using a single Tesla V100 GPU. The dev set performance during the training shows the exact slot matching performance, including ordering; it will range between 55 to 57 % for the final model.

The required files of the SPIDER dataset are: ```tables.json```, ```train_spider.json```, ```train_others.json```, plus ```dev.json``` for testing. 

## Evaluate 
Run:

```
python src/actual_test.py [BERT_DIR] [SPIDER_DATASET_DIR] [OUT_FILE]
```

to get the resultant SQL statements for the development set. The generated output file then could be evaluated using the SPIDER's evaluation script.

The performance of evaluation script with the final model will range from 64 to 66 %, since the ordering of conditions is not important for an actual SQL statement. 

The required files for SPIDER dataset is, ```table.json``` for database schema information, and ```dev.json``` for development dataset.
