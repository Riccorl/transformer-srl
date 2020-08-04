import os
import pathlib
from collections import defaultdict
from typing import Dict, List, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import (
    get_lengths_from_binary_sequence_mask,
    viterbi_decode,
    get_device_of,
)
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp_models.structured_prediction import SrlBert
from allennlp_models.structured_prediction.metrics.srl_eval_scorer import (
    DEFAULT_SRL_EVAL_PATH,
    SrlEvalScorer,
)
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from overrides import overrides
from torch.nn.modules import Linear, Dropout
from transformers import AutoModel

from transformer_srl.utils import load_lemma_frame, load_role_frame

LEMMA_FRAME_PATH = pathlib.Path(__file__).resolve().parent / "resources" / "lemma2frame.csv"
FRAME_ROLE_PATH = pathlib.Path(__file__).resolve().parent / "resources" / "frame2role.csv"


@Model.register("transformer_srl_dependency")
class TransformerSrlDependency(Model):
    """

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model : `Union[str, AutoModel]`, required.
        A string describing the BERT model to load or an already constructed AutoModel.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    label_smoothing : `float`, optional (default = `0.0`)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric : `bool`, optional (default = `False`)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    srl_eval_path : `str`, optional (default=`DEFAULT_SRL_EVAL_PATH`)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If `None`, srl-eval.pl is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        model_name: Union[str, AutoModel],
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        restrict_frames: bool = False,
        restrict_roles: bool = False,
        **kwargs,
    ) -> None:
        # bypass SrlBert constructor
        Model.__init__(self, vocab, **kwargs)
        self.lemma_frame_dict = load_lemma_frame(LEMMA_FRAME_PATH)
        self.frame_role_dict = load_role_frame(FRAME_ROLE_PATH)
        self.restrict_frames = restrict_frames
        self.restrict_roles = restrict_roles

        if isinstance(model_name, str):
            self.transformer = AutoModel.from_pretrained(model_name)
        else:
            self.transformer = model_name
        # loss
        self.role_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.frame_criterion = torch.nn.CrossEntropyLoss()
        # number of classes
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.frame_num_classes = self.vocab.get_vocab_size("frames_labels")
        # metrics
        role_set = self.vocab.get_token_to_index_vocabulary("labels")
        role_set_filter = [v for k, v in role_set.items() if k != "O"]
        self.f1_role_metric = FBetaMeasure(average="micro", labels=role_set_filter)
        self.f1_frame_metric = FBetaMeasure(average="micro")
        # output layer
        self.tag_projection_layer = Linear(self.transformer.config.hidden_size, self.num_classes)
        self.frame_projection_layer = Linear(
            self.transformer.config.hidden_size, self.frame_num_classes
        )
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        verb_indicator: torch.Tensor,
        frame_indicator: torch.Tensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
        frame_tags: torch.LongTensor = None,
    ):

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: `torch.LongTensor`, required.
            An integer `SequenceFeatureField` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels
            of shape `(batch_size, num_tokens)`
        frame_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the gold frames
            of shape ``(batch_size, num_tokens)``
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

        # Returns

        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            a distribution of the tag classes per word.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.transformer(
            input_ids=util.get_token_ids_from_text_field_tensors(tokens),
            token_type_ids=verb_indicator,
            attention_mask=mask,
        )

        # extract embeddings
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        frame_embeddings = embedded_text_input[frame_indicator == 1]
        # get sizes
        batch_size, sequence_length, _ = embedded_text_input.size()
        # outputs
        logits = self.tag_projection_layer(embedded_text_input)
        frame_logits = self.frame_projection_layer(frame_embeddings)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        role_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )
        frame_probabilities = F.softmax(frame_logits, dim=-1)
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict = {
            "logits": logits,
            "frame_logits": frame_logits,
            "role_probabilities": role_probabilities,
            "frame_probabilities": frame_probabilities,
            "mask": mask,
        }
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
        lemmas = [l for x in metadata for l in x["lemmas"]]
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["lemma"] = list(lemmas)

        if tags is not None:
            # compute role loss
            # role_loss = sequence_cross_entropy_with_logits(
            #     logits, tags, mask, label_smoothing=self._label_smoothing
            # )
            role_loss = self.role_criterion(logits.view(-1, self.num_classes), tags.view(-1))
            # compute frame loss
            frame_tags_filtered = frame_tags[frame_indicator == 1]
            frame_loss = self.frame_criterion(frame_logits, frame_tags_filtered)

            self.f1_role_metric(role_probabilities, tags)
            self.f1_frame_metric(frame_logits, frame_tags_filtered)

            output_dict["frame_loss"] = frame_loss
            output_dict["role_loss"] = role_loss
            output_dict["loss"] = (role_loss + frame_loss) / 2
        return output_dict

    def decode_frames(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # frame prediction
        frame_probabilities = output_dict["frame_probabilities"]
        if self.restrict:
            frame_probabilities = frame_probabilities.cpu().data.numpy()
            lemmas = output_dict["lemma"]
            candidate_labels = [self.lemma_frame_dict.get(l, []) for l in lemmas]
            # clear candidates from unknowns
            label_set = set(k for k in self._get_label_tokens("frames_labels"))
            candidate_labels_ids = [
                [
                    self.vocab.get_token_index(l, namespace="frames_labels")
                    for l in cl
                    if l in label_set
                ]
                for cl in candidate_labels
            ]

            frame_predictions = []
            for cl, fp in zip(candidate_labels_ids, frame_probabilities):
                # restrict candidates from verbatlas inventory
                fp_candidates = np.take(fp, cl)
                if fp_candidates.size > 0:
                    frame_predictions.append(cl[fp_candidates.argmax(axis=-1)])
                else:
                    frame_predictions.append(fp.argmax(axis=-1))
        else:
            frame_predictions = frame_probabilities.argmax(dim=-1).cpu().data.numpy()

        output_dict["frame_tags"] = [
            self.vocab.get_token_from_index(f, namespace="frames_labels") for f in frame_predictions
        ]
        output_dict["frame_scores"] = [
            fp[f] for f, fp in zip(frame_predictions, frame_probabilities)
        ]
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor], restrict: bool = True
    ) -> Dict[str, torch.Tensor]:
        output_dict = self.decode_frames(output_dict)
        # if self.restrict:
        #     output_dict = self._mask_args(output_dict)
        # output_dict = super().make_output_human_readable(output_dict)
        roles_probabilities = output_dict["role_probabilities"]
        roles_predictions = roles_probabilities.argmax(dim=-1).cpu().data.numpy()

        output_dict["tags"] = [
            [self.vocab.get_token_from_index(r, namespace="labels") for r in roles]
            for roles in roles_predictions
        ]
        return output_dict

    def _mask_args(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probs = output_dict["class_probabilities"]
        device = get_device_of(class_probs)
        lemmas = output_dict["lemma"]
        frames = output_dict["frame_tags"]
        candidate_mask = torch.ones_like(class_probs, dtype=torch.bool).to(device)
        for i, (l, f) in enumerate(zip(lemmas, frames)):
            candidates = self.frame_role_dict.get((l, f), [])
            if candidates:
                canidate_ids = [
                    self.vocab.get_token_index(r, namespace="labels") for r in candidates
                ]
                canidate_ids = torch.tensor(canidate_ids).to(device)
                canidate_ids = canidate_ids.repeat(candidate_mask.shape[1], 1)
                candidate_mask[i].scatter_(1, canidate_ids, False)
            else:
                candidate_mask[i].fill_(False)
        class_probs.masked_fill_(candidate_mask, 0)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        role_metric_dict = self.f1_role_metric.get_metric(reset=reset)
        frame_metric_dict = self.f1_frame_metric.get_metric(reset=reset)
        # This can be a lot of metrics, as there are 3 per class.
        # we only really care about the overall metrics, so we filter for them here.
        # metric_dict_filtered = {
        #     x.split("-")[0] + "_role": y for x, y in metric_dict.items() if "overall" in x
        # }
        metric_dict = {
            "f1_role": role_metric_dict["fscore"],
            "f1_frame": frame_metric_dict["fscore"],
        }
        return metric_dict

    def _get_label_tokens(self, namespace: str = "labels"):
        return self.vocab.get_token_to_index_vocabulary(namespace).keys()

    def _get_label_ids(self, namespace: str = "labels"):
        return self.vocab.get_index_to_token_vocabulary(namespace).keys()

    default_predictor = "transformer_srl"


@Model.register("transformer_srl_span")
class TransformerSrlSpan(SrlBert):
    """

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model : `Union[str, AutoModel]`, required.
        A string describing the BERT model to load or an already constructed AutoModel.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    label_smoothing : `float`, optional (default = `0.0`)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric : `bool`, optional (default = `False`)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    srl_eval_path : `str`, optional (default=`DEFAULT_SRL_EVAL_PATH`)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If `None`, srl-eval.pl is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, AutoModel],
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        restrict_frames: bool = False,
        restrict_roles: bool = False,
        **kwargs,
    ) -> None:
        # bypass SrlBert constructor
        Model.__init__(self, vocab, **kwargs)
        self.lemma_frame_dict = load_lemma_frame(LEMMA_FRAME_PATH)
        self.frame_role_dict = load_role_frame(FRAME_ROLE_PATH)
        self.restrict_frames = restrict_frames
        self.restrict_roles = restrict_roles

        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model
        self.frame_criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.frame_num_classes = self.vocab.get_vocab_size("frames_labels")
        if srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=["V"])
        else:
            self.span_metric = None
        self.f1_frame_metric = FBetaMeasure(average="micro")
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)
        self.frame_projection_layer = Linear(
            self.bert_model.config.hidden_size, self.frame_num_classes
        )
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        verb_indicator: torch.Tensor,
        frame_indicator: torch.Tensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
        frame_tags: torch.LongTensor = None,
    ):

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: `torch.LongTensor`, required.
            An integer `SequenceFeatureField` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        frame_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the frame
            in the sentence. This should have shape (batch_size, num_tokens). Similar to verb_indicator,
            but handles bert wordpiece tokenizer by cosnidering a frame only the first subtoken.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels
            of shape `(batch_size, num_tokens)`
        frame_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the gold frames
            of shape ``(batch_size, num_tokens)``
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

        # Returns

        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            a distribution of the tag classes per word.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        mask = get_text_field_mask(tokens)
        input_ids = util.get_token_ids_from_text_field_tensors(tokens)
        bert_embeddings, _ = self.bert_model(
            input_ids=input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            attention_mask=mask,
        )

        # extract embeddings
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        frame_embeddings = embedded_text_input[frame_indicator == 1]
        # get sizes
        batch_size, sequence_length, _ = embedded_text_input.size()
        # outputs
        logits = self.tag_projection_layer(embedded_text_input)
        frame_logits = self.frame_projection_layer(frame_embeddings)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )
        frame_probabilities = F.softmax(frame_logits, dim=-1)
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict = {
            "logits": logits,
            "frame_logits": frame_logits,
            "class_probabilities": class_probabilities,
            "frame_probabilities": frame_probabilities,
            "mask": mask,
        }
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        lemmas = [l for x in metadata for l in x["lemmas"]]
        output_dict["words"] = list(words)
        output_dict["lemma"] = list(lemmas)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if tags is not None:
            # compute role loss
            role_loss = sequence_cross_entropy_with_logits(
                logits, tags, mask, label_smoothing=self._label_smoothing
            )
            # compute frame loss
            frame_tags_filtered = frame_tags[frame_indicator == 1]
            frame_loss = self.frame_criterion(frame_logits, frame_tags_filtered)
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_verb_indices = [
                    example_metadata["verb_index"] for example_metadata in metadata
                ]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get the BIO tags from make_output_human_readable()
                batch_bio_predicted_tags = self.make_output_human_readable(output_dict).pop("tags")
                from allennlp_models.structured_prediction.models.srl import (
                    convert_bio_tags_to_conll_format,
                )

                batch_conll_predicted_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                ]
                batch_bio_gold_tags = [
                    example_metadata["gold_tags"] for example_metadata in metadata
                ]
                batch_conll_gold_tags = [
                    convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                ]
                self.span_metric(
                    batch_verb_indices,
                    batch_sentences,
                    batch_conll_predicted_tags,
                    batch_conll_gold_tags,
                )
            self.f1_frame_metric(frame_logits, frame_tags_filtered)
            output_dict["frame_loss"] = frame_loss
            output_dict["role_loss"] = role_loss
            output_dict["loss"] = (role_loss + frame_loss) / 2
        return output_dict

    def decode_frames(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # frame prediction
        frame_probabilities = output_dict["frame_probabilities"]
        if self.restrict_frames:
            frame_probabilities = frame_probabilities.cpu().data.numpy()
            lemmas = output_dict["lemma"]
            candidate_labels = [self.lemma_frame_dict.get(l, []) for l in lemmas]
            # clear candidates from unknowns
            label_set = set(k for k in self._get_label_tokens("frames_labels"))
            candidate_labels_ids = [
                [
                    self.vocab.get_token_index(l, namespace="frames_labels")
                    for l in cl
                    if l in label_set
                ]
                for cl in candidate_labels
            ]

            frame_predictions = []
            for cl, fp in zip(candidate_labels_ids, frame_probabilities):
                # restrict candidates from verbatlas inventory
                fp_candidates = np.take(fp, cl)
                if fp_candidates.size > 0:
                    frame_predictions.append(cl[fp_candidates.argmax(axis=-1)])
                else:
                    frame_predictions.append(fp.argmax(axis=-1))
        else:
            frame_predictions = frame_probabilities.argmax(dim=-1).cpu().data.numpy()

        output_dict["frame_tags"] = [
            self.vocab.get_token_from_index(f, namespace="frames_labels") for f in frame_predictions
        ]
        output_dict["frame_scores"] = [
            fp[f] for f, fp in zip(frame_predictions, frame_probabilities)
        ]
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output_dict = self.decode_frames(output_dict)
        if self.restrict_roles:
            output_dict = self._mask_args(output_dict)
        output_dict = super().make_output_human_readable(output_dict)
        return output_dict

    def _mask_args(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        class_probs = output_dict["class_probabilities"]
        device = get_device_of(class_probs)
        lemmas = output_dict["lemma"]
        frames = output_dict["frame_tags"]
        candidate_mask = torch.ones_like(class_probs, dtype=torch.bool).to(device)
        for i, (l, f) in enumerate(zip(lemmas, frames)):
            candidates = self.frame_role_dict.get((l, f), [])
            if candidates:
                canidate_ids = [
                    self.vocab.get_token_index(r, namespace="labels") for r in candidates
                ]
                canidate_ids = torch.tensor(canidate_ids).to(device)
                canidate_ids = canidate_ids.repeat(candidate_mask.shape[1], 1)
                candidate_mask[i].scatter_(1, canidate_ids, False)
            else:
                candidate_mask[i].fill_(False)
        class_probs.masked_fill_(candidate_mask, 0)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            frame_metric_dict = self.f1_frame_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            metric_dict_filtered = {
                x.split("-")[0] + "_role": y for x, y in metric_dict.items() if "overall" in x
            }
            frame_metric_dict = {x + "_frame": y for x, y in frame_metric_dict.items()}
            return {**metric_dict_filtered, **frame_metric_dict}

    def _get_label_tokens(self, namespace: str = "labels"):
        return self.vocab.get_token_to_index_vocabulary(namespace).keys()

    def _get_label_ids(self, namespace: str = "labels"):
        return self.vocab.get_index_to_token_vocabulary(namespace).keys()

    default_predictor = "transformer_srl"
