#!/bin/bash
source /home/pollo/miniconda3/bin/activate srl-mt

# local
DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-5.0-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/bert_tiny_span.jsonnet"

allennlp train "$CONFIG" -s models/test_span --include-package transformer_srl --recover
