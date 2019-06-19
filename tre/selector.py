import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Selector(nn.Module):

    def __init__(self, model, n_class, dropout):
        super(Selector, self).__init__()
        relation_dim = model.embed.embedding_dim

        self.relation_matrix = nn.Embedding(n_class, relation_dim)
        self.bias = nn.Parameter(torch.Tensor(n_class))
        #self.attention_matrix = nn.Embedding(n_class, relation_dim)
        self.init_weights()
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.xavier_uniform(self.relation_matrix.weight.data)
        nn.init.normal(self.bias)
        #nn.init.xavier_uniform(self.attention_matrix.weight.data)

    def get_logits(self, x):
        logits = torch.matmul(x, torch.transpose(
            self.relation_matrix.weight, 0, 1),) + self.bias
        return logits

    def forward(self, x, scopes=None, label=None):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError


class Average(Selector):

    def forward(self, x, scopes=None, label=None):
        scopes = scopes or [(0, x.size(0))]

        tower_repre = []
        for start, end in scopes:
            sen_matrix = x[start: end]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits


class Attention(Selector):
    def _attention_train_logit(self, x, query):
        relation_query = self.relation_matrix(query)
        attention_logit = torch.sum(x * relation_query, 1, True)
        return attention_logit
	
    def _attention_test_logit(self, x):
        attention_logit = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1))
        return attention_logit
	
    def forward(self, x, scopes=None, label=None):
        scopes = scopes or [(0, x.size(0))]

        if self.training:
            attention_logit = self._attention_train_logit(x, label)

            tower_repre = []
            for start, end in scopes:
                sen_matrix = x[start: end]
                attention_score = F.softmax(torch.transpose(attention_logit[start: end], 0, 1), 1)
                final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
                tower_repre.append(final_repre)
            stack_repre = torch.stack(tower_repre)
            stack_repre = self.dropout(stack_repre)
            logits = self.get_logits(stack_repre)
            return logits

        else:
            attention_logit = self._attention_test_logit(x)

            tower_output = []
            for start, end in scopes:
                sen_matrix = x[start: end]
                attention_score = F.softmax(torch.transpose(attention_logit[start: end], 0, 1), 1)
                final_repre = torch.matmul(attention_score, sen_matrix)
                logits = self.get_logits(final_repre)
                tower_output.append(torch.diag(F.softmax(logits, 1)))
            stack_output = torch.stack(tower_output)
            return stack_output
        
