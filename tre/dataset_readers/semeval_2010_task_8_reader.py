from typing import Dict, List, Tuple
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, SpanField
from allennlp.data.tokenizers.token import Token
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.word_splitter import OpenAISplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SemEval2010Task8Reader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as semeval_file:
            logger.info("Reading SemEval 2010 Task 8 instances from jsonl dataset at: %s", file_path)
            for line in semeval_file:
                example = json.loads(line)

                tokens = example["tokens"]
                label = example["label"]
                entity_indices = example["entities"]
                
                start_e1, end_e1 = entity_indices[0]
                start_e2, end_e2 = entity_indices[1]
                entity_1 = (start_e1, end_e1 - 1)
                entity_2 = (start_e2, end_e2 - 1)

                yield self.text_to_instance(tokens, entity_1, entity_2, label)
        
    @overrides
    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         entity_1: Tuple[int],
                         entity_2: Tuple[int],
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        tokens = [OpenAISplitter._standardize(token) for token in tokens]
        tokens = ['__start__'] + tokens[entity_1[0]:entity_1[1]+1] + ['__del1__'] + tokens[entity_2[0]:entity_2[1]+1] + ['__del2__'] + tokens + ['__clf__']
            
        sentence = TextField([Token(text=t) for t in tokens], self._token_indexers)
        fields['sentence'] = sentence
        #fields['entity1'] = SpanField(*entity_1, sequence_field=sentence)
        #fields['entity2'] = SpanField(*entity_2, sequence_field=sentence)
        
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)
