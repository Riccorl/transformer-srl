![Upload Python Package](https://github.com/Riccorl/srl-bert-verbatlas/workflows/Upload%20Python%20Package/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Semantic Role Lableing with BERT

Semantic Role Labeling based on [AllenNLP implementation](https://demo.allennlp.org/semantic-role-labeling) of [Shi et al, 2019](https://arxiv.org/abs/1904.05255). Can be trained using both PropBank and [VerbAatlas](http://verbatlas.org/) inventories and implements also the predicate disambiguation task, in addition to arguments identification and disambiguation.

### To-Dos

- [x] Works with both PropBank and VerbAtlas (infer inventory from dataset reader)
- [ ] Compatibility with all models from Huggingface's Transformers.
        - Now works only with models that accept 1 as token type id 
- [ ] Predicate identification (without using spacy)

### Infos

- Language Model: BERT
- Dataset: CoNLL 2012


### Results with VerbAtlas

With `bert-base-cased`:
```
# Dev set
- F1 arguments 87.6
- F1 predicates 95.5
# Test set
- F1 arguments x
- F1 predicates x
```

With `bert-base-multilingual-cased`:
```
# Dev set
- F1 arguments 86.2
- F1 predicates 94.2
# Test set
- F1 arguments 86.1
- F1 predicates 94.9
```
