#!/bin/bash
source /Users/ric/mambaforge/bin/activate srl-mt

#HOME="/home/orlando"
DATASET="/Users/ric/Documents/ComputerScience/Projects/transformer-srl/data/conll2012_pb_subset/"
PROJECT="/Users/ric/Documents/ComputerScience/Projects/transformer-srl"
# local
# DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
# PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/srl-bert-span"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/bert_base_span.jsonnet"
MODEL_DIR="$PROJECT/models/bert_base_conll2012"

allennlp train $CONFIG -s $MODEL_DIR --include-package transformer_srl --force #--recover
