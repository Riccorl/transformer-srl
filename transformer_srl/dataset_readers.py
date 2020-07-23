import logging
from typing import Dict, List, Iterable, Tuple, Any

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence
from allennlp_models.structured_prediction import SrlReader
from overrides import overrides
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]):
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    # Parameters

    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    # Returns

    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


def _convert_frames_indices_to_wordpiece_indices(
    frame_labels: List[int], offsets: List[int], binary: bool = False
):
    """
    Converts frame labels to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.
    This is only used if you pass a `bert_model_name` to the dataset reader below.
    # Parameters
    frame_labels : `List[int]`
        Frame labels.
    offsets : `List[int]`
        The wordpiece offsets.
    # Returns
    The new frame labels.
    """
    j = 0
    new_frame_labels = []
    for i, offset in enumerate(offsets):
        indicator = frame_labels[i]
        if indicator != "O" and indicator != 0:
            new_frame_labels.append(indicator)
            j += 1
            indicator = 0 if binary else "O"
        while j < offset:
            new_frame_labels.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    if binary:
        return [0] + new_frame_labels + [0]
    else:
        return ["O"] + new_frame_labels + ["O"]


@DatasetReader.register("transformer_srl")
class SrlTransformersReader(SrlReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : `TextField`
        The tokens in the sentence.
    verb_indicator : `SequenceLabelField`
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : `SequenceLabelField`
        A sequence of Propbank tags for the given verb in a BIO format.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is `{"tokens": PretrainedTransformerIndexer()}`.
    domain_identifier : `str`, (default = `None`)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : `Optional[str]`, (default = `None`)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.

    # Returns

    A `Dataset` of `Instances` for Semantic Role Labelling.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        domain_identifier: str = None,
        bert_model_name: str = None,
        **kwargs,
    ) -> None:
        DatasetReader.__init__(self, **kwargs)
        self._token_indexers = token_indexers or {"tokens": PretrainedTransformerIndexer(bert_model_name)}
        self._domain_identifier = domain_identifier

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.lowercase_input = "uncased" in bert_model_name

    def _wordpiece_tokenize_input(
        self, tokens: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        # Returns

        wordpieces : `List[str]`
            The BERT wordpieces from the words in the sentence.
        end_offsets : `List[int]`
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : `List[int]`
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """
        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = (
            [self.bert_tokenizer.cls_token] + word_piece_tokens + [self.bert_tokenizer.sep_token]
        )
        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(
                "Filtering to only include file paths containing the %s domain",
                self._domain_identifier,
            )

        for sentence in self._ontonotes_subset(
            ontonotes_reader, file_path, self._domain_identifier
        ):
            tokens = [Token(t) for t in sentence.words]
            if sentence.srl_frames:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    frames = [
                        f if v == 1 else "O"
                        for f, v in zip(sentence.predicate_framenet_ids, verb_indicator)
                    ]
                    lemmas = [
                        f for f, v in zip(sentence.predicate_lemmas, verb_indicator) if v == 1
                    ]
                    if not all(v == 0 for v in verb_indicator):
                        yield self.text_to_instance(tokens, verb_indicator, frames, lemmas, tags)

    def text_to_instance(  # type: ignore
        self,
        tokens: List[Token],
        verb_label: List[int],
        frames: List[str] = None,
        lemmas: List[str] = None,
        tags: List[str] = None,
    ) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """

        metadata_dict: Dict[str, Any] = {}
        wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(
            [t.text for t in tokens]
        )
        new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
        frame_indicator = _convert_frames_indices_to_wordpiece_indices(verb_label, offsets, True)
        metadata_dict["offsets"] = start_offsets
        # In order to override the indexing mechanism, we need to set the `text_id`
        # attribute directly. This causes the indexing to use this id.
        text_field = TextField(
            [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
            token_indexers=self._token_indexers,
        )
        verb_indicator = SequenceLabelField(new_verbs, text_field)
        frame_indicator = SequenceLabelField(frame_indicator, text_field)

        fields: Dict[str, Field] = {
            "tokens": text_field,
            "verb_indicator": verb_indicator,
            "frame_indicator": frame_indicator,
        }

        if all(x == 0 for x in verb_label):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text

        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["lemmas"] = lemmas
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
            new_frames = _convert_frames_indices_to_wordpiece_indices(frames, offsets)
            fields["tags"] = SequenceLabelField(new_tags, text_field)
            fields["frame_tags"] = SequenceLabelField(
                new_frames, text_field, label_namespace="frames_labels"
            )
            metadata_dict["gold_tags"] = tags
            metadata_dict["gold_frame_tags"] = frames

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
