from typing import List

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
from overrides import overrides
from spacy.tokens import Doc
from allennlp.models.archival import Archive, load_archive


@Predictor.register("srl_verbatlas")
class SRL(SemanticRoleLabelerPredictor):
    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm",
    ) -> None:
        super().__init__(model, dataset_reader, language)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def predict_tokenized(self, tokenized_sentence: List[str]) -> JsonDict:
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)

        tokens = [token for token in spacy_doc]
        instances = self.tokens_to_instances(tokens)

        if not instances:
            return sanitize({"verbs": [], "lemmas": [], "poses": [], "words": tokens})

        return self.predict_instances(instances)

    @staticmethod
    def make_srl_string(words: List[str], tags: List[str], frame: str) -> str:
        window = []
        chunk = []

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    window.append("[" + " ".join(chunk) + "]")
                    chunk = []

                if tag.startswith("B-"):
                    tag = tag.replace("V", frame)
                    chunk.append(tag[2:] + ": " + token)
                elif tag == "O":
                    window.append(token)

        if chunk:
            window.append("[" + " ".join(chunk) + "]")

        return " ".join(window)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL model uses a different API for creating instances.")

    def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            if word.pos_ == "VERB":
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self.tokens_to_instances(tokens)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        # For SRL, we have more instances than sentences, but the user specified
        # a batch size with respect to the number of sentences passed, so we respect
        # that here by taking the batch size which we use to be the number of sentences
        # we are given.
        batch_size = len(inputs)
        instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]

        flattened_instances = [
            instance
            for sentence_instances in instances_per_sentence
            for instance in sentence_instances
        ]

        if not flattened_instances:
            output = [
                {
                    "verbs": [],
                    "lemmas": [],
                    "poses": [],
                    "words": self._tokenizer.split_words(x["sentence"]),
                }
                for x in inputs
            ]
            return sanitize(output)

        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [
            instance for instance in batched_instances[-1] if instance is not None
        ]
        # Run the model on the batches.
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))

        verbs_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts: List[JsonDict] = [{"verbs": [], "lemmas": [], "poses": [],} for x in inputs]

        output_index = 0
        for sentence_index, verb_count in enumerate(verbs_per_sentence):
            if verb_count == 0:
                # We didn't run any predictions for sentences with no verbs,
                # so we don't have a way to extract the original sentence.
                # Here we just tokenize the input again.
                original_text = self._tokenizer.split_words(inputs[sentence_index]["sentence"])
                return_dicts[sentence_index]["words"] = original_text
                continue

            for _ in range(verb_count):
                output = outputs[output_index]
                words = output["words"]
                tags = output["tags"]
                frame = output["frame_tags"]
                description = self.make_srl_string(words, tags, frame)
                return_dicts[sentence_index]["words"] = words
                return_dicts[sentence_index]["poses"] = output["poses"]
                return_dicts[sentence_index]["words"] = output["lemmas"]
                return_dicts[sentence_index]["verbs"].append(
                    {
                        "verb": output["verb"],
                        "description": description,
                        "tags": tags,
                        "frame": frame,
                    }
                )
                output_index += 1

        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)

        results = {
            "verbs": [],
            "words": outputs[0]["words"],
            "poses": outputs[0]["poses"],
            "lemmas": outputs[0]["lemmas"],
        }
        for output in outputs:
            words = output["words"]
            tags = output["tags"]
            frame = output["frame_tags"]
            description = self.make_srl_string(words, tags, frame)
            results["verbs"].append(
                {"verb": output["verb"], "description": description, "tags": tags, "frame": frame,}
            )

        return sanitize(results)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instances = self._sentence_to_srl_instances(inputs)

        if not instances:
            return sanitize(
                {
                    "verbs": [],
                    "lemmas": [],
                    "poses": [],
                    "words": self._tokenizer.split_words(inputs["sentence"]),
                }
            )

        return self.predict_instances(instances)

    @classmethod
    def from_path(
        cls,
        archive_path: str,
        predictor_name: str = None,
        cuda_device: int = -1,
        language: str = "en_core_web_sm",
        dataset_reader_to_load: str = "validation",
    ) -> "SRL":
        """
        Instantiate a :class:`Predictor` from an archive path.

        If you need more detailed configuration options, such as overrides,
        please use `from_archive`.

        Parameters
        ----------
        archive_path: ``str``
            The path to the archive.
        predictor_name: ``str``, optional (default=None)
            Name that the predictor is registered as, or None to use the
            predictor associated with the model.
        cuda_device: ``int``, optional (default=-1)
            If `cuda_device` is >= 0, the model will be loaded onto the
            corresponding GPU. Otherwise it will be loaded onto the CPU.
        dataset_reader_to_load: ``str``, optional (default="validation")
            Which dataset reader to load from the archive, either "train" or
            "validation".

        Returns
        -------
        A Predictor instance.
        """
        return SRL.from_archive(
            load_archive(archive_path, cuda_device=cuda_device),
            predictor_name,
            language=language,
            dataset_reader_to_load=dataset_reader_to_load,
        )

    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: str = None,
        language: str = "en_core_web_sm",
        dataset_reader_to_load: str = "validation",
    ) -> "SRL":
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used. Optionally specify
        which :class:`DatasetReader` should be loaded; otherwise, the validation one will be used
        if it exists followed by the training dataset reader.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("type")
            if not model_type in DEFAULT_PREDICTORS:
                raise ConfigurationError(
                    f"No default predictor for model type {model_type}.\n"
                    f"Please specify a predictor explicitly."
                )
            predictor_name = DEFAULT_PREDICTORS[model_type]

        if dataset_reader_to_load == "validation" and "validation_dataset_reader" in config:
            dataset_reader_params = config["validation_dataset_reader"]
        else:
            dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        model.eval()

        return Predictor.by_name(predictor_name)(model, dataset_reader, language)
