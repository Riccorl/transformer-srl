#!/bin/bash

HOME="/content"
DATASET="$HOME/conll-formatted-ontonotes-verbatlas"
PROJECT="$HOME/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/data/train"
export SRL_VALIDATION_DATA_PATH="$DATASET/data/development"

CONFIG="$PROJECT/training_config/bert_base_ml.jsonnet"

MODEL_FOLDER="/content/drive/My Drive/Sapienza/NLPLab/models/allennlp/bert_base_ml"
allennlp train $CONFIG -s "$MODEL_FOLDER" --include-package transformer_srl
