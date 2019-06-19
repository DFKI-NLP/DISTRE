from typing import Dict, List, Any, Tuple

from overrides import overrides
import torch

from torch import nn
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders.openai_transformer_embedder import OpenaiTransformerEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
from tre.transformer import OpenaiTransformer
from tre.selector import Average, Attention


class TaskHead(nn.Module):
    pass


class LanguageModelHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model):
        super(LanguageModelHead, self).__init__()
        self.n_embd = model.embed.embedding_dim
        self.decoder = model.decoder

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(h_trunc)
        return lm_logits
    
    
class ClassificationHead(TaskHead):
    def __init__(self, model, n_class, clf_token, dropout=0.):
        super(ClassificationHead, self).__init__()
        self.n_embd = model.embed.embedding_dim
        self.clf_token = clf_token
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.n_embd, n_class)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.contiguous().view(-1, self.n_embd)
        flat = x.contiguous().view(-1)
        clf_h = clf_h[flat == self.clf_token, :]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)

        return clf_logits


class TRE(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 vocab: Vocabulary,
                 lm_head: LanguageModelHead=None,
                 clf_head: ClassificationHead=None,
                 language_model_weight: float=.5) -> None:
        
        super().__init__(vocab)
        
        assert not (lm_head is None and clf_head is None)
        
        self.embedder = embedder
        self.clf_head = clf_head
        self.lm_head = lm_head
        self.language_model_weight = language_model_weight
        self.vocab = vocab
        
    def _loss(self, sentence: Dict[str, torch.Tensor], sentence_encoded: torch.Tensor, label: torch.Tensor):
        lm_losses = 0.
        clf_losses = 0.

        # [n_batch, seq_len]
        byte_pairs = sentence['byte_pairs'][:, :sentence_encoded.size(1)]
        
        if self.lm_head is not None:
            # [n_batch, seq_len]
            mask = (byte_pairs != 0).float()
            
            # (n_batch, seq_len - 1)
            lm_logits = self.lm_head(sentence_encoded)
            #print('lm_logits:', lm_logits.shape)
            
            # [n_batch * (seq_len - 1)]
            byte_pairs_shifted = byte_pairs[:, 1:].contiguous().view(-1)
            
            # [n_batch * (seq_len - 1)]
            lm_losses = nn.CrossEntropyLoss(reduce=False)(lm_logits, byte_pairs_shifted)
            
            # [n_batch, seq_len - 1]
            lm_losses = lm_losses.view(byte_pairs.size(0), byte_pairs.size(1) - 1)
            
            # [n_batch, seq_len - 1]
            lm_losses = lm_losses * mask[:, 1:]
            
            # [n_batch * (seq_len - 1)]
            lm_losses = lm_losses.sum(dim=1) / torch.sum(mask[:, 1:], dim=1)
            lm_losses = lm_losses.mean()
            
        if self.clf_head is not None:
            clf_logits = self.clf_head(sentence_encoded, byte_pairs)
            if clf_logits.size(0) != label.size(0):
                torch.save(sentence, './sentence.pt')
            clf_losses = nn.CrossEntropyLoss(reduce=False)(clf_logits, label)
            clf_losses = clf_losses.mean()
            
        loss = clf_losses + self.language_model_weight * lm_losses
            
        return loss
        
        
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        
        sentence_encoded = self.embedder(sentence['byte_pairs'])

        output = {}
        
        if label is not None:
            output["loss"] = self._loss(sentence, sentence_encoded, label)
        
        if self.clf_head is not None:
            byte_pairs = sentence['byte_pairs'][:, :sentence_encoded.size(1)]
            clf_logits = self.clf_head(sentence_encoded, byte_pairs)
            output['logits'] = clf_logits
        
        return output


from operator import itemgetter
from itertools import groupby

def get_scopes(x):
    scopes = []
    for k, g in groupby(enumerate(x), lambda i_x: i_x[1]):
        group = list(map(itemgetter(0), g))
        scopes.append((group[0], group[-1]+1))
    return scopes


def get_entity_mask(x, del1, del2):
    a = torch.arange(x.size(1)).expand_as(x).type_as(x)
    pos_del1 = a[x == del1].unsqueeze(1)
    pos_del2 = a[x == del2].unsqueeze(1)
    mask_ent1 = (0 < a) & (a < pos_del1)
    mask_ent2 = (pos_del1 < a) & (a < pos_del2)
    return mask_ent1, mask_ent2


def get_entity_dropout_mask(x, del1, del2, dropout_probability=.1):
    mask_ent1, mask_ent2 = get_entity_mask(x, del1, del2)
    binary_mask_ent1 = x.new(x.size(0), 1).bernoulli_(1 - dropout_probability)
    binary_mask_ent2 = x.new(x.size(0), 1).bernoulli_(1 - dropout_probability)
    return (binary_mask_ent1 * mask_ent1.long() + binary_mask_ent2 * mask_ent2.long()).byte()


class BagClassificationHead(TaskHead):
    def __init__(self, model, n_class, encoder_vocab, clf_token, selector, dropout=0.):
        super(BagClassificationHead, self).__init__()
        self.n_embd = model.embed.embedding_dim
        self.encoder_vocab = encoder_vocab
        self.clf_token = clf_token

        if selector == 'average':
            self.selector = Average(model, n_class, dropout)
        elif selector == 'attention':
            self.selector = Attention(model, n_class, dropout)
        else:
            raise ValueError(f"Selector '{selector}' not supported.")

    def forward(self, sentence: Dict[str, torch.Tensor], sentence_encoded: torch.Tensor, scopes: List[Tuple[int, int]], label: torch.Tensor):
        x = sentence['byte_pairs'][:, :sentence_encoded.size(1)]
        h = sentence_encoded

        clf_token_idx = self.encoder_vocab[self.clf_token]

        clf_h = h.contiguous().view(-1, self.n_embd)
        flat = x.contiguous().view(-1)
        clf_h = clf_h[flat == clf_token_idx, :]
        clf_logits = self.selector(clf_h, scopes, label)

        return clf_logits, clf_h


@Model.register("mitre")
class MITRE(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 openai_model_path: str,
                 n_ctx: int=512,
                 tokens_to_add: List[str]=None,
                 requires_grad: bool=True,
                 clf_token: str='__clf__',
                 dropout: float=.1,
                 entity_dropout: float=0.0,
                 language_model_weight: float=.5,
                 selector: str='average',
                 label_namespace='labels') -> None:
        
        super().__init__(vocab)

        n_special = len(tokens_to_add) if tokens_to_add is not None else -1

        transformer = OpenaiTransformer(
            model_path=openai_model_path,
            n_special=n_special,
            requires_grad=requires_grad,
            n_ctx=n_ctx)

        self.embedder = OpenaiTransformerEmbedder(transformer=transformer, top_layer_only=True)
        
        self.clf_head = BagClassificationHead(
            model=transformer,
            encoder_vocab=vocab.get_token_to_index_vocabulary('openai_transformer'),
            n_class=vocab.get_vocab_size(label_namespace),
            clf_token=clf_token + '</w>',
            selector=selector,
            dropout=dropout)

        self.lm_head = LanguageModelHead(transformer)
        self.language_model_weight = language_model_weight

        self.entity_dropout = entity_dropout

        self.encoder_vocab = vocab.get_token_to_index_vocabulary('openai_transformer')
        self.del1_token = '__del1__</w>'
        self.del2_token = '__del2__</w>'
        self.mask_token = '__mask__</w>'
        self.na_idx = self.vocab.get_token_to_index_vocabulary('labels')['NA']

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "not_na_accuracy": CategoricalAccuracy()
        }
        
    def _loss(self, sentence: Dict[str, torch.Tensor], sentence_encoded: torch.Tensor, scopes: List[Tuple[int, int]], label: torch.Tensor):
        lm_losses = 0.
        clf_losses = 0.

        bag_label_indices = [start for start, _ in scopes]
        bag_label = label[bag_label_indices]

        # [n_batch, seq_len]
        byte_pairs = sentence['byte_pairs'][:, :sentence_encoded.size(1)]
        
        if self.lm_head is not None:
            # [n_batch, seq_len]
            mask = (byte_pairs != 0).float()
            
            # (n_batch, seq_len - 1)
            lm_logits = self.lm_head(sentence_encoded)
            
            # [n_batch * (seq_len - 1)]
            byte_pairs_shifted = byte_pairs[:, 1:].contiguous().view(-1)
            
            # [n_batch * (seq_len - 1)]
            lm_losses = nn.CrossEntropyLoss(reduce=False)(lm_logits, byte_pairs_shifted)
            
            # [n_batch, seq_len - 1]
            lm_losses = lm_losses.view(byte_pairs.size(0), byte_pairs.size(1) - 1)
            
            # [n_batch, seq_len - 1]
            lm_losses = lm_losses * mask[:, 1:]
            
            # [n_batch * (seq_len - 1)]
            lm_losses = lm_losses.sum(dim=1) / torch.sum(mask[:, 1:], dim=1)
            lm_losses = lm_losses.mean()
            
        if self.clf_head is not None:
            clf_logits, _ = self.clf_head(sentence, sentence_encoded, scopes, label=label)
            clf_losses = nn.CrossEntropyLoss(reduce=False)(clf_logits, bag_label)
            clf_losses = clf_losses.mean()
            
        loss = clf_losses + self.language_model_weight * lm_losses

        self.metrics['accuracy'](clf_logits, bag_label)
        self.metrics['not_na_accuracy'](clf_logits, bag_label, mask=torch.ne(bag_label, self.na_idx))
            
        return loss
        
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                metadata: List[Dict[str, Any]],
                label: torch.Tensor = None) -> torch.Tensor:
        
        byte_pairs = sentence['byte_pairs']

        if self.training and self.entity_dropout > 0.:
            dropout_mask = get_entity_dropout_mask(byte_pairs, self.encoder_vocab[self.del1_token], self.encoder_vocab[self.del2_token], self.entity_dropout)
            byte_pairs = byte_pairs.masked_fill(dropout_mask, self.encoder_vocab[self.mask_token])

        sentence_encoded = self.embedder(byte_pairs)
        
        output = {}

        instance_ids = [md['instance_id'] for md in metadata]
        scopes = get_scopes(instance_ids)
        
        if label is not None:
            output["loss"] = self._loss(sentence, sentence_encoded, scopes, label)
        
        clf_logits, clf_h = self.clf_head(sentence, sentence_encoded, scopes, label)
        output['instances'] = [metadata[idx] for idx, _ in scopes]
        output['logits'] = clf_logits
        output['clf_h'] = clf_h
        
        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
