#!/bin/bash
source /home/pollo/miniconda3/bin/activate allennlp

#HOME="/home/orlando"
#DATASET="$HOME/datasets/ontonotes/conll-formatted-ontonotes-verbatlas"
#PROJECT="$HOME/srl-bert-span"
# local
DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="/mnt/d/Datasets/conll2012/conll_like/un/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/bert_base_ml.jsonnet"

allennlp train "$CONFIG" -s models/bert_base_ml_un --include-package transformer_srl --recover
