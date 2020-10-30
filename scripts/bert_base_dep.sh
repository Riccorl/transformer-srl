#!/bin/bash
source /home/pollo/miniconda3/bin/activate srl-mt

# HOME="/home/orlando"
# DATASET="$HOME/datasets/ontonotes/conll-formatted-ontonotes-verbatlas"
# local
DATASET="/mnt/d/Datasets/UniversalProposition/UniversalPropositions-conllus-format/UP_Italian"
PROJECT="/mnt/c/Users/rikkw/Desktop/Ric/Projects/transformer-srl"

#export SRL_TRAIN_DATA_PATH="$DATASET/it-up-train.conllus"
export SRL_TRAIN_DATA_PATH="/mnt/d/Datasets/Wikimatrix/it/wikimatrix.en-it.it.direct.conllus"
export SRL_VALIDATION_DATA_PATH="$DATASET/it-up-dev.conllus"

CONFIG="$PROJECT/training_config/bert_base_dep.jsonnet"

rm -r models/test_dep
allennlp train $CONFIG -s models/test_dep --include-package transformer_srl #--recover
