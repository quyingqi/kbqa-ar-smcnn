#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-12-12
import sys, os
import pickle
import numpy as np
import random
import torch
from torch.autograd import Variable
sys.path.append('../vocab')
sys.path.append('../tools')
import virtuoso
from args import get_args
args = get_args()

def create_seq_ranking_data(qa_data, word_vocab, rel_sep_vocab, rel_vocab, save_path):
    seqs = []
    seq_len = []
    pos_rel1 = []
    pos_rel2 = []
    neg_rel1 = []
    neg_rel2 = []
    pos_rel = []
    neg_rel = []
    pos_rel_len = []
    neg_rel_len = []
    batch_index = -1    # the index of sequence batches
    seq_index = 0       # sequence index within each batch
    pad_index = word_vocab.lookup(word_vocab.pad_token)

    rel_max_len = args.rel_maxlen

    data_list = pickle.load(open(qa_data, 'rb'))

    def get_separated_rel_id(relation):
        rel = relation.split('.')
        rel1 = '.'.join(rel[:-1])
        rel2 = rel[-1]
        rel_word = []
        rel[0] = rel[0][3:]
        for i in rel:
            rel_word.extend(i.split('_'))

        rel1_id = rel_sep_vocab[0].convert_to_index([rel1])[0]
        rel2_id = rel_sep_vocab[1].convert_to_index([rel2])[0]
        rel_id = word_vocab.convert_to_index(rel_word)
        return rel1_id, rel2_id, rel_id

    for data in data_list:
        tokens = data.question_pattern.split()
        can_rels = []
        if hasattr(data, 'cand_sub') and data.subject in data.cand_sub:
            can_rels = data.cand_rel
        else:
            can_subs = virtuoso.str_query_id(data.text_subject)
            for sub in can_subs:
                can_rels.extend(virtuoso.id_query_out_rel(sub))
            can_rels = list(set(can_rels))
        if data.relation in can_rels:
            can_rels.remove(data.relation)
        for i in range(len(can_rels), args.neg_size):
            tmp = random.randint(2, len(rel_vocab)-1)
            while(tmp in can_rels):
                tmp = random.randint(2, len(rel_vocab)-1)
            can_rels.append(rel_vocab.index2word[tmp])
        can_rels = random.sample(can_rels, args.neg_size)

        if seq_index % args.batch_size == 0:
            seq_index = 0
            batch_index += 1
            seqs.append(torch.LongTensor(args.batch_size, len(tokens)).fill_(pad_index))
            seq_len.append(torch.LongTensor(args.batch_size).fill_(1))
            pos_rel1.append(torch.LongTensor(args.batch_size).fill_(pad_index))
            pos_rel2.append(torch.LongTensor(args.batch_size).fill_(pad_index))
            neg_rel1.append(torch.LongTensor(args.neg_size, args.batch_size).fill_(pad_index))
            neg_rel2.append(torch.LongTensor(args.neg_size, args.batch_size).fill_(pad_index))
            pos_rel.append(torch.LongTensor(args.batch_size, rel_max_len).fill_(pad_index))
            pos_rel_len.append(torch.Tensor(args.batch_size).fill_(1))
            neg_rel.append(torch.LongTensor(args.neg_size, args.batch_size, rel_max_len).fill_(pad_index))
            neg_rel_len.append(torch.Tensor(args.neg_size, args.batch_size).fill_(1))
            print('batch: %d' %batch_index)

        seqs[batch_index][seq_index, 0:len(tokens)] = torch.LongTensor(word_vocab.convert_to_index(tokens))
        seq_len[batch_index][seq_index] = len(tokens)

        pos1, pos2, pos = get_separated_rel_id(data.relation)
        pos_rel1[batch_index][seq_index] = pos1
        pos_rel2[batch_index][seq_index] = pos2
        pos_rel[batch_index][seq_index, 0:len(pos)] = torch.LongTensor(pos)
        pos_rel_len[batch_index][seq_index] = len(pos)

        for j, can_rel in enumerate(can_rels):
            neg1, neg2, neg = get_separated_rel_id(can_rel)
            if not neg1 or not neg2:
                continue
            neg_rel1[batch_index][j,seq_index] = neg1
            neg_rel2[batch_index][j,seq_index] = neg2
            neg_rel[batch_index][j,seq_index, 0:len(neg)] = torch.LongTensor(neg)
            neg_rel_len[batch_index][j,seq_index] = len(neg)

        seq_index += 1

    torch.save((seqs, seq_len, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, pos_rel_len, neg_rel, neg_rel_len), save_path)

class SeqRankingLoader():
    def __init__(self, infile, device=-1):
        self.seqs, self.seq_len, self.pos_rel1, self.pos_rel2, self.neg_rel1, self.neg_rel2, self.pos_rel, self.pos_rel_len, self.neg_rel, self.neg_rel_len = torch.load(infile)
        self.batch_size = self.seqs[0].size(0)
        self.batch_num = len(self.seqs)

        if device >=0:
            for i in range(self.batch_num):
                self.seqs[i] = self.seqs[i].cuda(device)
                self.pos_rel1[i] = self.pos_rel1[i].cuda(device)
                self.pos_rel2[i] = self.pos_rel2[i].cuda(device)
                self.neg_rel1[i] = self.neg_rel1[i].cuda(device)
                self.neg_rel2[i] = self.neg_rel2[i].cuda(device)
                self.pos_rel[i] = self.pos_rel[i].cuda(device)
                self.neg_rel[i] = self.neg_rel[i].cuda(device)

    def next_batch(self, shuffle = True):
        if shuffle:
            indices = torch.randperm(self.batch_num)
        else:
            indices = range(self.batch_num)
        for i in indices:
            yield Variable(self.seqs[i]), self.seq_len[i], Variable(self.pos_rel1[i]), \
            Variable(self.pos_rel2[i]), Variable(self.neg_rel1[i]), Variable(self.neg_rel2[i]), \
            Variable(self.pos_rel[i]), self.pos_rel_len[i], Variable(self.neg_rel[i]),\
            self.neg_rel_len[i]

class CandidateRankingLoader():
    def __init__(self, qa_pattern_file, word_vocab, rel_sep_vocab, device=-1):
        self.qa_pattern = pickle.load(open(qa_pattern_file, 'rb'))
        self.batch_num = len(self.qa_pattern)
        self.word_vocab = word_vocab
        self.rel_sep_vocab = rel_sep_vocab
        self.pad_index = word_vocab.lookup(word_vocab.pad_token)
        self.device = device

    def get_separated_rel_id(self, relation):
        rel = relation.split('.')
        rel1 = '.'.join(rel[:-1])
        rel2 = rel[-1]
        rel_word = []
        rel[0] = rel[0][3:]
        for i in rel:
            rel_word.extend(i.split('_'))

        rel1_id = self.rel_sep_vocab[0].convert_to_index([rel1])[0]
        rel2_id = self.rel_sep_vocab[1].convert_to_index([rel2])[0]
        rel_id = self.word_vocab.convert_to_index(rel_word)
        return rel1_id, rel2_id, rel_id

    def next_question(self):
        for data in self.qa_pattern:
            if not hasattr(data, 'cand_rel'):
                self.batch_num -= 1
                continue

            tokens = data.question_pattern.split()
            seqs = torch.LongTensor(self.word_vocab.convert_to_index(tokens)).unsqueeze(0)
            seq_len = torch.LongTensor([len(tokens)])

            pos1, pos2, pos = self.get_separated_rel_id(data.relation)
            pos_rel1 = torch.LongTensor([pos1])
            pos_rel2 = torch.LongTensor([pos2])
            pos_rel = torch.LongTensor(args.rel_maxlen).fill_(self.pad_index)
            pos_rel[0:len(pos)] = torch.LongTensor(pos)
            pos_rel = pos_rel.unsqueeze(0)
            pos_len = torch.LongTensor([len(pos)])

            neg_rel1 = torch.LongTensor(len(data.cand_rel))
            neg_rel2 = torch.LongTensor(len(data.cand_rel))
            neg_rel = torch.LongTensor(len(data.cand_rel), args.rel_maxlen).fill_(self.pad_index)
            neg_len = torch.LongTensor(len(data.cand_rel))
            for idx, rel in enumerate(data.cand_rel):
                neg1, neg2, neg = self.get_separated_rel_id(rel)
                neg_rel1[idx] = neg1
                neg_rel2[idx] = neg2
                neg_rel[idx, 0:len(neg)] = torch.LongTensor(neg)
                neg_len[idx] = len(neg)
            neg_rel1.unsqueeze_(1)
            neg_rel2.unsqueeze_(1)
            neg_rel.unsqueeze_(1)

            if self.device>=0:
                seqs, pos_rel1, pos_rel2, neg_rel1, neg_rel2, pos_rel, neg_rel = \
                seqs.cuda(self.device), pos_rel1.cuda(self.device), pos_rel2.cuda(self.device), \
                neg_rel1.cuda(self.device), neg_rel2.cuda(self.device), pos_rel.cuda(self.device), \
                neg_rel.cuda(self.device)
            yield Variable(seqs), seq_len, Variable(pos_rel1), Variable(pos_rel2), Variable(neg_rel1), Variable(neg_rel2), Variable(pos_rel), pos_len, Variable(neg_rel), neg_len, data

if __name__ == '__main__':
    word_vocab = torch.load(args.vocab_file)
    rel_sep_vocab = torch.load(args.rel_vocab_file)
    rel_vocab = torch.load('../vocab/vocab.rel.pt')

    qa_data_path = '../entity_detection/results/QAData.label.%s.pkl'
    if not os.path.exists('data'):
        os.mkdir('data')

    create_seq_ranking_data(qa_data_path % 'valid', word_vocab, rel_sep_vocab, rel_vocab, args.valid_file)
    create_seq_ranking_data(qa_data_path % 'test', word_vocab, rel_sep_vocab, rel_vocab, args.test_file)
    create_seq_ranking_data(qa_data_path % 'train', word_vocab, rel_sep_vocab, rel_vocab, args.train_file)
