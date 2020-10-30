#!/bin/bash
source /home/pollo/miniconda3/bin/activate srl-mt-rc

# local
# DATASET="/mnt/d/Datasets/conll2012/conll-formatted-ontonotes-verbatlas-subset"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

# export SRL_TRAIN_DATA_PATH="$DATASET/train"
# export SRL_VALIDATION_DATA_PATH="$DATASET/development"
# export SRL_TRAIN_DATA_PATH="/mnt/d/Datasets/UN/en/train"
export SRL_TRAIN_DATA_PATH="/mnt/d/Datasets/semeval2021/test/fr"
export SRL_VALIDATION_DATA_PATH="/mnt/d/Datasets/semeval2021/test"

# CONFIG="$PROJECT/training_config/bert_tiny_span.jsonnet"
CONFIG="$PROJECT/training_config/xlm_r_base_span.jsonnet"
# CONFIG="$PROJECT/training_config/bert_base_span.jsonnet"

rm -r models/test_xlmr

allennlp train "$CONFIG" -s models/test_xlmr --include-package transformer_srl #--recover
