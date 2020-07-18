#!/bin/bash
#source /home/orlando/miniconda3/bin/activate allennlp-09
source /home/pollo/miniconda3/bin/activate allennlp-test

#HOME="/home/orlando"
#DATASET="$HOME/datasets/ontonotes/conll-formatted-ontonotes-verbatlas"
#PROJECT="$HOME/srl-bert-span"
# local
DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/srl-bert-verbatlas"

export SRL_TRAIN_DATA_PATH="/mnt/d/Datasets/conll2012/conll_like/un/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/srl_bert_base_ml.jsonnet"

allennlp train "$CONFIG" -s models/srl_bert_base_ml_un --include-package srl_bert_verbatlas --recover
