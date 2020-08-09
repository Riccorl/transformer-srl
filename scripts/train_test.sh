#!/bin/bash
source /Users/ric/miniconda3/bin/activate srl-mt

# local
DATASET="data/semeval/"
PROJECT="/Users/ric/Documents/ComputerScience/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET"
export SRL_VALIDATION_DATA_PATH="$DATASET"

CONFIG="$PROJECT/training_config/bert_tiny_span.jsonnet"

allennlp train "$CONFIG" -s models/test_span --include-package transformer_srl #--recover
