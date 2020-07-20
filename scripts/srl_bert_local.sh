#!/bin/bash
source /home/pollo/miniconda3/bin/activate allennlp

# HOME="/home/orlando"
# DATASET="$HOME/datasets/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0"
# PROJECT="$HOME/srl-bert-span"
# local
DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/srl-bert-verbatlas"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/srl_bert_tiny.jsonnet"

allennlp train $CONFIG -s models/srl_transformers --include-package srl_transformers # srl_verbatlas
