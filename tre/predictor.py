from typing import Tuple, List

from overrides import overrides

import torch
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.predictors.predictor import Predictor


@Predictor.register('tre-classifier')
class TREClassifierPredictor(Predictor):
    """Predictor wrapper for the TREClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict['sentence']
        head = json_dict['head']
        tail = json_dict['tail']
        instance = self._dataset_reader.text_to_instance(sentence=sentence, head=head, tail=tail)
        return instance

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        model = self._model

        with torch.no_grad():
            cuda_device = model._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = model.decode(model(**model_input))

        return sanitize(outputs)
