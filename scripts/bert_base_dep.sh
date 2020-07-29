#!/bin/bash
source /home/pollo/miniconda3/bin/activate allennlp-test

# HOME="/home/orlando"
# DATASET="$HOME/datasets/ontonotes/conll-formatted-ontonotes-verbatlas"
# local
DATASET="/mnt/d/Datasets/UniversalProposition/english"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

export SRL_TRAIN_DATA_PATH="$DATASET/en_ewt-up-train.conllu"
export SRL_VALIDATION_DATA_PATH="$DATASET/en_ewt-up-dev.conllu"

CONFIG="$PROJECT/training_config/dep_bert_base.jsonnet"

allennlp train $CONFIG -s models/dep_bert_base --include-package transformer_srl #--recover
