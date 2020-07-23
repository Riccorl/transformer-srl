#!/bin/bash
source /home/pollo/miniconda3/bin/activate allennlp

# local
DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/bert_tiny.jsonnet"

allennlp train $CONFIG -s models/transformers_local --include-package transformer_srl
