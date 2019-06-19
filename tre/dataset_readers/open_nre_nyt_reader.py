from typing import Dict, List, Tuple
import json
import logging
from itertools import groupby

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.tokenizers.token import Token
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.word_splitter import OpenAISplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("open_nre_nyt_reader")
class OpenNreNYTReader(DatasetReader):
    def __init__(self,
                 masking_mode: str=None,
                 token_indexers: Dict[str, TokenIndexer]=None,
                 lazy: bool=False) -> None:
        super().__init__(lazy)

        if masking_mode and masking_mode.lower() not in ['ner_least_specific', 'ner_most_specific']:
            raise ValueError(f"Masking mode '{masking_mode}' not supported.")

        self._masking_mode = masking_mode
        self._token_splitter = OpenAISplitter()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        
    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'rb') as f:
            nyt_dataset = json.load(f)

            for mention in nyt_dataset:
                sentence = mention['sentence']
                head = mention['head']['word']
                tail = mention['tail']['word']
                relation = mention['relation']

                head_type = None
                tail_type = None

                if self._masking_mode == 'ner_least_specific':
                    head_types = mention['head']['corrected_type']
                    tail_types = mention['tail']['corrected_type']

                    if head_types:
                        head_type = list(sorted(head_types, key=lambda t: t.count('/')))[0]
                    else:
                        head_type = 'n/a'
                    
                    if tail_types:
                        tail_type = list(sorted(tail_types, key=lambda t: t.count('/')))[0]
                    else:
                        head_type = 'n/a'

                    head_type = '__' + head_type + '__'
                    tail_type = '__' + tail_type + '__'

                yield self.text_to_instance(sentence=sentence, head=head, tail=tail, label=relation,
                                            head_type=head_type, tail_type=tail_type)
    
    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: str,
                         head: str,
                         tail: str,
                         head_type: str=None,
                         tail_type: str=None,
                         label: str=None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        
        instance_id = f'{head}#{tail}'
        if label:
            instance_id = f'{instance_id}#{label}'

        fields['metadata'] = MetadataField({'instance_id': instance_id.lower()})

        tokens = self._token_splitter.split_words(sentence)
        head = self._token_splitter.split_words(head)
        tail = self._token_splitter.split_words(tail)

        # TODO: this should not be done here

        if self._masking_mode == 'ner_least_specific':
            logger.info(f"Using masking mode 'ner_least_specific'.")
            tokens = ([Token('__start__')]
                      + head + [Token('__del1__')] + head_type + [Token('__ent1__')]
                      + tail + [Token('__del2__')] + tail_type + [Token('__ent2__')]
                      + tokens + [Token('__clf__')])
        else:
            tokens = [Token('__start__')] + head + [Token('__del1__')] + tail + [Token('__del2__')] + tokens + [Token('__clf__')]

        fields['sentence'] = TextField(tokens, self._token_indexers)
        
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)
