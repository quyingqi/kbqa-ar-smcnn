#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-15

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import sys
sys.path.append('../tools')
from embedding import Embeddings
from attention import MLPWordSeqAttention

class RelationRanking(nn.Module):

    def __init__(self, word_vocab, rel_vocab, config):
        super(RelationRanking, self).__init__()
        self.config = config
        rel1_vocab, rel2_vocab = rel_vocab
        self.word_embed = Embeddings(word_vec_size=config.d_word_embed, dicts=word_vocab)
        self.rel1_embed = Embeddings(word_vec_size=config.d_rel_embed, dicts=rel1_vocab)
        self.rel2_embed = Embeddings(word_vec_size=config.d_rel_embed, dicts=rel2_vocab)

        if self.config.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                              num_layers=config.n_layers, dropout=config.dropout_prob,
                              bidirectional=config.birnn,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=config.d_word_embed, hidden_size=config.d_hidden,
                               num_layers=config.n_layers, dropout=config.dropout_prob,
                               bidirectional=config.birnn,
                               batch_first=True)

        self.dropout = nn.Dropout(p=config.dropout_prob)
        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2

        self.question_attention = MLPWordSeqAttention(input_size=config.d_rel_embed, seq_size=seq_in_size)

        self.bilinear = nn.Bilinear(seq_in_size, config.d_rel_embed, 1, bias=False)

        self.seq_out = nn.Sequential(
                        self.dropout,
                        nn.Linear(seq_in_size, config.d_rel_embed)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, config.channel_size, (config.conv_kernel_1, config.conv_kernel_2), stride=1,
                      padding=(config.conv_kernel_1//2, config.conv_kernel_2//2)), #channel_in=1, channel_out=8, kernel_size=3*3
            nn.ReLU(True))

        self.seq_maxlen = config.seq_maxlen + (config.conv_kernel_1 + 1) % 2
        self.rel_maxlen = config.rel_maxlen + (config.conv_kernel_2 + 1) % 2

        self.pooling = nn.MaxPool2d((config.seq_maxlen, 1),
                                    stride=(config.seq_maxlen, 1), padding=0)

        self.pooling2 = nn.MaxPool2d((1, config.rel_maxlen),
                                    stride=(1, config.rel_maxlen), padding=0)

        self.fc = nn.Sequential(
            nn.Linear(config.rel_maxlen * config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(20, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(config.seq_maxlen * config.channel_size, 20),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob),
            nn.Linear(20,1))

        self.fc2 = nn.Sequential(
            nn.Linear(4, 1))


    def question_encoder(self, inputs):
        '''
        :param inputs: (batch, dim1)
        '''
        batch_size = inputs.size(0)
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        if self.config.rnn_type.lower() == 'gru':
            h0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, ht = self.rnn(inputs, h0)
        else:
            h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
            outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        # shape of `outputs` - (batch size, sequence length, hidden size X num directions)
        outputs.contiguous()
        return outputs

    def cal_score(self, outputs, seqs_len, rel_embed, pos=None):
        '''
        :param rel_embed: (batch, dim2) or (neg_size, batch, dim2)
        return: (batch, 1)
        '''
        batch_size = outputs.size(0)
        if pos:
            neg_size = pos
        else:
            neg_size, batch_size, embed_size = rel_embed.size()
            seq_len, seq_emb_size = outputs.size()[1:]
            outputs = outputs.unsqueeze(0).expand(neg_size, batch_size, seq_len,
                            seq_emb_size).contiguous().view(neg_size*batch_size, seq_len, -1)
            rel_embed = rel_embed.view(neg_size * batch_size, -1)
            seqs_len = seqs_len.unsqueeze(0).expand(neg_size, batch_size).contiguous().view(neg_size*batch_size)
        # `weight` - (batch, length)
        seq_att, weight = self.question_attention.forward(rel_embed, outputs)
        # `seq_encode` - (batch, hidden size X num directions)
        seq_encode = self.seq_out(seq_att)

        # `score` - (batch, 1) or (neg_size * batch, 1)
        score = torch.sum(seq_encode * rel_embed, 1, keepdim=True)

        if pos:
            score = score.unsqueeze(0).expand(neg_size, batch_size, 1)
        else:
            score = score.view(neg_size, batch_size, 1)
        return score

    def matchPyramid(self, seq, rel, seq_len, rel_len):
        '''
        param:
            seq: (batch, _seq_len, embed_size)
            rel: (batch, _rel_len, embed_size)
            seq_len: (batch,)
            rel_len: (batch,)
        return:
            score: (batch, 1)
        '''
        batch_size = seq.size(0)

        rel_trans = torch.transpose(rel, 1, 2)
        # (batch, 1, seq_len, rel_len)
        seq_norm = torch.sqrt(torch.sum(seq*seq, dim=2, keepdim=True))
        rel_norm = torch.sqrt(torch.sum(rel_trans*rel_trans, dim=1, keepdim=True))
        cross = torch.bmm(seq/seq_norm, rel_trans/rel_norm).unsqueeze(1)

        # (batch, channel_size, seq_len, rel_len)
        conv1 = self.conv(cross)
        channel_size = conv1.size(1)

        # (batch, seq_maxlen)
        # (batch, rel_maxlen)
        dpool_index1, dpool_index2 = self.dynamic_pooling_index(seq_len, rel_len, self.seq_maxlen,
                                                                self.rel_maxlen)
        dpool_index1 = dpool_index1.unsqueeze(1).unsqueeze(-1).expand(batch_size, channel_size,
                                                                self.seq_maxlen, self.rel_maxlen)
        dpool_index2 = dpool_index2.unsqueeze(1).unsqueeze(2).expand_as(dpool_index1)
        conv1_expand = torch.gather(conv1, 2, dpool_index1)
        conv1_expand = torch.gather(conv1_expand, 3, dpool_index2)

        # (batch, channel_size, p_size1, p_size2)
        pool1 = self.pooling(conv1_expand).view(batch_size, -1)

        # (batch, 1)
        out = self.fc(pool1)

        pool2 = self.pooling2(conv1_expand).view(batch_size, -1)
        out2 = self.fc1(pool2)

        return out, out2

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            return idx1_one, idx2_one
        batch_size = len(len1)
        index1, index2 = [], []
        for i in range(batch_size):
            idx1_one, idx2_one = dpool_index_(i, len1[i], len2[i], max_len1, max_len2)
            index1.append(idx1_one)
            index2.append(idx2_one)
        index1 = torch.LongTensor(index1)
        index2 = torch.LongTensor(index2)
        if self.config.cuda:
            index1 = index1.cuda()
            index2 = index2.cuda()
        return Variable(index1), Variable(index2)


    def forward(self, batch):
        # shape of seqs (batch size, sequence length)
        seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, pos_rel_len, neg_rel, neg_rel_len = batch

        # shape (batch_size, sequence length, dimension of embedding)
        inputs = self.word_embed.forward(seqs)
        outputs = self.question_encoder(inputs)

        # shape (batch_size, dimension of rel embedding)
        pos_rel1_embed = self.rel1_embed.word_lookup_table(pos_rel1)
        pos_rel2_embed = self.rel2_embed.word_lookup_table(pos_rel2)
        pos_rel1_embed = self.dropout(pos_rel1_embed)
        pos_rel2_embed = self.dropout(pos_rel2_embed)
        # shape (neg_size, batch_size, dimension of rel embedding)
        neg_rel1_embed = self.rel1_embed.word_lookup_table(neg_rel1)
        neg_rel2_embed = self.rel2_embed.word_lookup_table(neg_rel2)
        neg_rel1_embed = self.dropout(neg_rel1_embed)
        neg_rel2_embed = self.dropout(neg_rel2_embed)

        neg_size, batch, neg_len = neg_rel.size()
        # shape of `score` - (neg_size, batch_size, 1)
        pos_score1 = self.cal_score(outputs, seq_len, pos_rel1_embed, neg_size)
        pos_score2 = self.cal_score(outputs, seq_len, pos_rel2_embed, neg_size)
        neg_score1 = self.cal_score(outputs, seq_len, neg_rel1_embed)
        neg_score2 = self.cal_score(outputs, seq_len, neg_rel2_embed)

        # (batch, len, emb_size)
        pos_embed = self.word_embed.forward(pos_rel)
        # (batch, 20)
        pos_score3, pos_score4 = self.matchPyramid(inputs, pos_embed, seq_len, pos_rel_len)
        # (neg_size, batch, 20)
        pos_score3 = pos_score3.unsqueeze(0).expand(neg_size, batch, pos_score3.size(1))
        pos_score4 = pos_score4.unsqueeze(0).expand(neg_size, batch, pos_score4.size(1))

        # (neg_size*batch, len, emb_size)
        neg_embed = self.word_embed.forward(neg_rel.view(-1, neg_len))
        seqs_embed = inputs.unsqueeze(0).expand(neg_size, batch, inputs.size(1),
                    inputs.size(2)).contiguous().view(-1, inputs.size(1), inputs.size(2))
        # (neg_size*batch,)
        neg_rel_len = neg_rel_len.view(-1)
        seq_len = seq_len.unsqueeze(0).expand(neg_size, batch).contiguous().view(-1)
        # (neg_size*batch, 20)
        neg_score3, neg_score4 = self.matchPyramid(seqs_embed, neg_embed, seq_len, neg_rel_len)
        # (neg_size, batch, 20)
        neg_score3 = neg_score3.view(neg_size, batch, neg_score3.size(1))
        neg_score4 = neg_score4.view(neg_size, batch, neg_score4.size(1))

        pos_concat = torch.cat((pos_score1, pos_score2, pos_score3, pos_score4), 2)
        neg_concat = torch.cat((neg_score1, neg_score2, neg_score3, neg_score4), 2)
        pos_score = self.fc2(pos_concat).squeeze(-1)
        neg_score = self.fc2(neg_concat).squeeze(-1)

        return pos_score, neg_score
