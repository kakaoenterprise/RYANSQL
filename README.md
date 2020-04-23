# RYANSQL
## Introduction
A source code for RYANSQL, a text-to-SQL system for complex, cross-domain databases.

Reference Paper: https://arxiv.org/abs/2004.03125

The system is submitted to SPIDER leaderboard(https://yale-lily.github.io/spider). The system and its minor improved version RYANSQL v2 is ranked at second and fourth place (as of February 2020).

The system does NOT use any database records, which make it more acceptable to the real world company applications. Among the systems not using the database records, the system is ranked #1 in the leaderboard.

## Requirements
Python3 <br>
Tensorflow 1.14 <br>
nltk

## Install 
Download the BERT model from https://github.com/google-research/bert. You can only download the model, not the whole git. The system uses BERT-large, uncased with Whole Word Masking model (https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip). Unzip the downloaded file.
 
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

The training takes about a day or 2 using a single Tesla V100 GPU. The development performance during the training shows the exact slot matching performance, including ordering; the actual performance of the final model will range between 64 to 66%, since the ordering of columns are not important in the final SQL statement.

The required files of the SPIDER dataset are: tables.json, train_spider.json, train_others.json, plus dev.json for testing. 

## Evaluate 
Run:

```

