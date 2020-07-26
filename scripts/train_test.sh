#!/bin/bash
source /Users/ric/miniconda3/bin/activate srl-mt

# local
DATASET="data/UP_English-EWT"
PROJECT="/Users/ric/Documents/ComputerScience/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/en_ewt-up-train.conllu"
export SRL_VALIDATION_DATA_PATH="$DATASET/en_ewt-up-dev.conllu"

CONFIG="$PROJECT/training_config/bert_tiny_dep.jsonnet"

allennlp train "$CONFIG" -s models/test_dep --include-package transformer_srl --recover
