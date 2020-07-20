import os
from collections import defaultdict
import pathlib
from typing import Dict, List, Optional, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.srl_util import convert_bio_tags_to_conll_format
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.srl_eval_scorer import (
    SrlEvalScorer,
    DEFAULT_SRL_EVAL_PATH,
)
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
from torch.nn.modules import Linear, Dropout


LEMMA_FRAME_PATH = pathlib.Path(__file__).resolve().parent.parent / "resources" / "lemma2frame.csv"


def read_dictionary(filename: pathlib.Path) -> Dict:
    """
    Open a dictionary from file, in the format key -> value
    :param filename: file to read.
    :return: a dictionary.
    """
    dictionary = defaultdict(list)
    with open(filename) as file:
        for l in file:
            k, *v = l.split()
            dictionary[k] += v
    return dictionary


@Model.register("srl_bert_verbatlas")
class SrlBertVerbatlas(Model):
    """

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    bert_model : ``Union[str, BertModel]``, required.
        A string describing the BERT model to load or an already constructed BertModel.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric: ``bool``, optional (default = False)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    srl_eval_path: ``str``, optional (default=``DEFAULT_SRL_EVAL_PATH``)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If ``None``, srl-eval.pl is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
    ) -> None:
        super(SrlBertVerbatlas, self).__init__(vocab, regularizer)
        self.lemma_frame_dict = read_dictionary(LEMMA_FRAME_PATH)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model
        self.frame_criterion = torch.nn.CrossEntropyLoss()
        # num classes
        self.num_classes = self.vocab.get_vocab_size("roles_labels")
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

    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.Tensor],
        verb_indicator: torch.Tensor,
        frame_indicator: torch.Tensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
        frame_tags: torch.LongTensor = None,
    ):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        frame_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the frame
            in the sentence. This should have shape (batch_size, num_tokens). Similar to verb_indicator,
            but handles bert wordpiece tokenizer by cosnidering a frame only the first subtoken.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        frame_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the gold frames
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(
            input_ids=tokens["tokens"],
            token_type_ids=verb_indicator,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )

        embedded_text_input = self.embedding_dropout(bert_embeddings)
        verbs_embeddings = embedded_text_input[frame_indicator == 1]
        batch_size, sequence_length, _ = embedded_text_input.size()
        logits = self.tag_projection_layer(embedded_text_input)
        frame_logits = self.frame_projection_layer(verbs_embeddings)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )
        frame_probabilities = F.softmax(frame_logits, dim=-1)
        output_dict = {
            "logits": logits,
            "frame_logits": frame_logits,
            "class_probabilities": class_probabilities,
            "frame_probabilities": frame_probabilities,
            "mask": mask,
        }
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        lemmas = [l for x in metadata for l in x["lemmas"]]
        output_dict["words"] = list(words)
        output_dict["lemmas"] = list(lemmas)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if tags is not None:
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
                # Get the BIO tags from decode()
                # TODO (nfliu): This is kind of a hack, consider splitting out part
                # of decode() to a separate function.
                batch_bio_predicted_tags = self.decode(output_dict, False).pop("tags")
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

    def decode_frames(
        self, output_dict: Dict[str, torch.Tensor], restrict: bool = True
    ) -> Dict[str, torch.Tensor]:
        # frame prediction
        frame_probabilities = output_dict["frame_probabilities"]
        if not restrict:
            frame_predictions = frame_probabilities.argmax(dim=-1).cpu().data.numpy()
            output_dict["frame_tags"] = [
                self.vocab.get_token_from_index(f, namespace="frames_labels")
                for f in frame_predictions
            ]
            return output_dict
        frame_probabilities = frame_probabilities.cpu().data.numpy()
        lemmas = output_dict["lemmas"]
        candidate_labels = [self.lemm_frame_dict.get(l, []) for l in lemmas]
        # clear candidates from unknowns
        label_set = set(k for k in self.vocab.get_token_to_index_vocabulary("frames_labels").keys())
        candidate_labels_ids = [
            self.vocab.get_token_index(l, namespace="frames_labels")
            for cl in candidate_labels
            for l in cl
            if l in label_set
        ]
        frame_predictions = []
        for cl, fp in zip(candidate_labels_ids, frame_probabilities):
            # restrict candidates from verbatlas inventory
            fp_candidates = np.take(fp, cl)
            if fp_candidates.size > 0:
                frame_predictions.append(fp_candidates.argmax(dim=-1))
            else:
                frame_predictions.append(fp.argmax(dim=-1))

        output_dict["frame_tags"] = [
            self.vocab.get_token_from_index(f, namespace="frames_labels") for f in frame_predictions
        ]
        return output_dict

    @overrides
    def decode(
        self, output_dict: Dict[str, torch.Tensor], restrict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.

        NOTE: First, we decode a BIO sequence on top of the wordpieces. This is important; viterbi
        decoding produces low quality output if you decode on top of word representations directly,
        because the model gets confused by the 'missing' positions (which is sensible as it is trained
        to perform tagging on wordpieces, not words).

        Secondly, it's important that the indices we use to recover words from the wordpieces are the
        start_offsets (i.e offsets which correspond to using the first wordpiece of words which are
        tokenized into multiple wordpieces) as otherwise, we might get an ill-formed BIO sequence
        when we select out the word tags from the wordpiece tags. This happens in the case that a word
        is split into multiple word pieces, and then we take the last tag of the word, which might
        correspond to, e.g, I-V, which would not be allowed as it is not preceeded by a B tag.
        """
        all_predictions = output_dict["class_probabilities"]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [
                all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))
            ]
        else:
            predictions_list = [all_predictions]
        wordpiece_tags = []
        word_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(
            predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length],
                transition_matrix,
                allowed_start_transitions=start_transitions,
            )
            tags = [
                self.vocab.get_token_from_index(x, namespace="roles_labels")
                for x in max_likelihood_sequence
            ]

            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
        output_dict["wordpiece_tags"] = wordpiece_tags
        output_dict["tags"] = word_tags
        self.decode_frames(output_dict, restrict)

        return output_dict

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

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("roles_labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        Returns
        -------
        start_transitions : torch.Tensor
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("roles_labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions
