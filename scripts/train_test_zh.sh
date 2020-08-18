#!/bin/bash
source /home/pollo/miniconda3/bin/activate srl-mt

# local
# D:\Datasets\conll2012\ric\conll2012-v4\conll-formatted-ontonotes-5.0\data\train\data
DATASET="/mnt/d/Datasets/conll2012/ric/conll2012-v4/conll-formatted-ontonotes-5.0"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train/data/chinese"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development/data/chinese"

CONFIG="$PROJECT/training_config/bert_tiny_span.jsonnet"

rm -r models/test_span_zh
allennlp train "$CONFIG" -s models/test_span_zh --include-package transformer_srl # --recover
