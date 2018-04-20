# -*- coding:utf-8 -*- 
import abc
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WordSeqAttentionModel(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_size, seq_size):
        super(WordSeqAttentionModel, self).__init__()
        self.input_size = input_size
        self.output_size = seq_size
        self.seq_size = seq_size

    @abc.abstractmethod
    def _score(self, x, seq):
        """
        Using through attention function
        :param x:
        :param seq:
        :return:
        """
        pass

    def attention(self, x, seq, lengths=None):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :return: weight: (batch, length)
        """
        # Check Size
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size

        score = self._score(x, seq)

        weight = F.softmax(score)
        return weight

    def forward(self, x, seq, lengths=None):
        """
        :param x: (batch, dim, )
        :param seq: (batch, length, dim, )
        :param lengths: (batch, )
        :return: hidden: (batch, dim)
                 weight: (batch, length)
        """
        # (batch, length)
        weight = self.attention(x, seq, lengths)
        # (batch, 1, length) bmm (batch, length, dim) -> (batch, 1, dim) -> (batch, dim)
        return torch.bmm(weight[:, None, :], seq).squeeze(1), weight

    def check_size(self, x, seq):
        batch_size, input_size = x.size()
        seq_batch_size, max_len, seq_size = seq.size()
        assert batch_size == seq_batch_size
        assert input_size == self.input_size
        assert seq_size == self.seq_size

    @staticmethod
    def expand_x(x, max_len):
        """
        :param x: (batch, input_size)
        :param max_len: scalar
        :return:  (batch * max_len, input_size)
        """
        batch_size, input_size = x.size()
        return torch.unsqueeze(x, 1).expand(batch_size, max_len, input_size).contiguous().view(batch_size * max_len, -1)

    @staticmethod
    def pack_seq(seq):
        """
        :param seq: (batch_size, max_len, seq_size)
        :return: (batch_size * max_len, seq_size)
        """
        return seq.view(seq.size(0) * seq.size(1), -1)

class MLPWordSeqAttention(WordSeqAttentionModel):
    def __init__(self, input_size, seq_size, hidden_size=None, activation="Tanh", bias=False):
        super(MLPWordSeqAttention, self).__init__(input_size=input_size, seq_size=seq_size)
        self.bias = bias
        self.hidden_size = hidden_size
        if hidden_size is None:
            hidden_size = (input_size + seq_size) // 2
        component = OrderedDict()
        component['layer1'] = nn.Linear(input_size + seq_size, hidden_size, bias=bias)
        component['act'] = getattr(nn, activation)()
        component['layer2'] = nn.Linear(hidden_size, 1, bias=bias)
        self.layer = nn.Sequential(component)

    def _score(self, x, seq):
        """
        :param x: (batch, word_dim)
        :param seq: (batch, length, seq_dim)
        :return: score: (batch, length, )
        """
        self.check_size(x, seq)

        # (batch, word_dim) -> (batch * max_len, word_dim)
        _x = self.expand_x(x, max_len=seq.size(1))

        # (batch, max_len, seq_dim) -> (batch * max_len, seq_dim)
        _seq = self.pack_seq(seq)

        # (batch * max_len, word_dim) (batch * max_len, seq_dim) -> (batch * max_len, word_dim + seq_dim)
        to_input = torch.cat([_x, _seq], 1)

        # (batch * max_len, word_dim + seq_dim)
        #   -> (batch * max_len, 1)
        #       -> (batch * max_len, )
        #           -> (batch, max_len)
        score = self.layer.forward(to_input).squeeze(-1).view(seq.size(0), seq.size(1))

        return score

