#!/bin/bash

export SRL_TRAIN_DATA_PATH="../input/conllchineseverbatlas/conll-chinese-verbatlas/train"
export SRL_VALIDATION_DATA_PATH="../input/unenzhzhbertgold-conll/un.en-zh.zh.bert.gold_conll"

allennlp train transformer-srl/training_config/bert_base_span.jsonnet -s "bert_zh_un" --include-package transformer_srl
