import enum
from typing import List, Dict, Type
from allennlp.data.tokenizers.token import Token

import numpy
from allennlp.common import plugins
from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.predictors.predictor import Predictor
from allennlp_models.structured_prediction import SemanticRoleLabelerPredictor
from overrides import overrides
from spacy.tokens import Doc


@Predictor.register("transformer_srl")
class SrlTransformersPredictor(SemanticRoleLabelerPredictor):
    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm",
    ) -> None:
        super().__init__(model, dataset_reader, language)

    @staticmethod
    def make_srl_string(words: List[str], tags: List[str], frame: str) -> str:
        srl_string = SemanticRoleLabelerPredictor.make_srl_string(words, tags)
        srl_string = srl_string.replace("[V", "[" + frame)
        return srl_string

    @overrides
    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        sentence = json_dict["sentence"]
        if json_dict.get("verbs"):
            text = sentence.split()
            pos = ["VERB" if i == json_dict["verbs"] else "NOUN" for i, _ in enumerate(text)]
            tokens = [Token(t, i, i + len(text), pos_=p) for i, (t, p) in enumerate(zip(text, pos))]
        else:
            tokens = self._tokenizer.tokenize(sentence)
        return self.tokens_to_instances(tokens)

    @overrides
    def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            if word.pos_ in ["AUX", "VERB"]:
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(
                    tokens, verb_labels, lemmas=[word.lemma_]
                )
                instances.append(instance)
        return instances

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
            return sanitize(
                [{"verbs": [], "words": self._tokenizer.tokenize(x["sentence"])} for x in inputs]
            )

        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [
            instance for instance in batched_instances[-1] if instance is not None
        ]
        # Run the model on the batches.
        outputs: List[Dict[str, numpy.ndarray]] = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))

        verbs_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts: List[JsonDict] = [{"verbs": []} for x in inputs]

        output_index = 0
        for sentence_index, verb_count in enumerate(verbs_per_sentence):
            if verb_count == 0:
                # We didn't run any predictions for sentences with no verbs,
                # so we don't have a way to extract the original sentence.
                # Here we just tokenize the input again.
                original_text = self._tokenizer.tokenize(inputs[sentence_index]["sentence"])
                return_dicts[sentence_index]["words"] = original_text
                continue

            for _ in range(verb_count):
                output = outputs[output_index]
                words = output["words"]
                tags = output["tags"]
                frame = output["frame_tags"]
                description = self.make_srl_string(words, tags, frame)
                return_dicts[sentence_index]["words"] = words
                verb_dict = {
                    "verb": output["verb"],
                    "description": description,
                    "tags": tags,
                    "frame": frame,
                    "frame_scores": output["frame_scores"],
                    "lemma": output["lemma"],
                }
                return_dicts[sentence_index]["verbs"].append(verb_dict)
                output_index += 1

        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)

        results = {"verbs": [], "words": outputs[0]["words"]}
        for output in outputs:
            tags = output["tags"]
            frame = output["frame_tags"]
            description = self.make_srl_string(output["words"], tags, frame)
            verb_dict = {
                "verb": output["verb"],
                "description": description,
                "tags": tags,
                "frame": frame,
                "frame_score": output["frame_scores"],
                "lemma": output["lemma"],
            }
            results["verbs"].append(verb_dict)

        return sanitize(results)

    @classmethod
    def from_path(
        cls,
        archive_path: str,
        predictor_name: str = None,
        cuda_device: int = -1,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        import_plugins: bool = True,
        language: str = "en_core_web_sm",
        restrict_frames: bool = False,
        restrict_roles: bool = False,
    ) -> "Predictor":
        if import_plugins:
            plugins.import_plugins()
        return SrlTransformersPredictor.from_archive(
            load_archive(archive_path, cuda_device=cuda_device),
            predictor_name,
            dataset_reader_to_load=dataset_reader_to_load,
            frozen=frozen,
            language=language,
            restrict_frames=restrict_frames,
            restrict_roles=restrict_roles,
        )

    @classmethod
    def from_archive(
        cls,
        archive: Archive,
        predictor_name: str = None,
        dataset_reader_to_load: str = "validation",
        frozen: bool = True,
        language: str = "en_core_web_sm",
        restrict_frames: bool = False,
        restrict_roles: bool = False,
    ) -> "Predictor":
        # Duplicate the config so that the config inside the archive doesn't get consumed
        config = archive.config.duplicate()

        if not predictor_name:
            model_type = config.get("model").get("type")
            model_class, _ = Model.resolve_class_name(model_type)
            predictor_name = model_class.default_predictor
        predictor_class: Type[Predictor] = Predictor.by_name(  # type: ignore
            predictor_name
        ) if predictor_name is not None else cls

        if dataset_reader_to_load == "validation" and "validation_dataset_reader" in config:
            dataset_reader_params = config["validation_dataset_reader"]
        else:
            dataset_reader_params = config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = archive.model
        if frozen:
            model.restrict_frames = restrict_frames
            model.restrict_roles = restrict_roles
            model.eval()

        return predictor_class(model, dataset_reader, language)
