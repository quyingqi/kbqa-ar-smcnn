#!/usr/bin/env python
#-*- coding: utf-8 -*-
 
# Author: QuYingqi
# mail: cookiequ17@hotmail.com
# Created Time: 2017-11-05

from dictionary import Dictionary
import torch
import pickle
import sys
sys.path.append('../tools')
from qa_data import QAData

def load_word_dictionary(filename, word_dict=None):
    if word_dict is None:
        word_dict = Dictionary()
        word_dict.add_unk_token()
        word_dict.add_pad_token()
    with open(filename) as f:
        for line in f:
            if not line:break
            line = line.strip()
            if not line:continue
            word_dict.add(line)
    return word_dict

def load_rel_separated_dictionary(filename):
    rel1_dict = Dictionary()
    rel1_dict.add_unk_token()
    rel1_dict.add_pad_token()
    rel2_dict = Dictionary()
    rel2_dict.add_unk_token()
    rel2_dict.add_pad_token()
    with open(filename) as f:
        for line in f:
            if not line:break
            line = line.strip()
            if not line:continue
            line = line.split('.')
            rel1 = '.'.join(line[:-1])
            rel2 = line[-1]
            rel1_dict.add(rel1)
            rel2_dict.add(rel2)
    return rel1_dict, rel2_dict

def creat_word_rel_dict(r_file, *q_files):
    word_dict = Dictionary()
    word_dict.add_unk_token()
    word_dict.add_pad_token()
    word_dict.add_start_token()

    for q_file in q_files:
        qa_data = pickle.load(open(q_file, 'rb'))
        for data in qa_data:
            q = data.question
            tokens = q.split(' ')
            for token in tokens:
                word_dict.add(token)
    print(len(word_dict))

    rels = pickle.load(open(r_file, 'rb'))
    for rel in rels:
        rel_word = []
        w = rel[3:].split('.')
        for i in w:
            rel_word.extend(i.split('_'))
        for word in rel_word:
            word_dict.add(word)
    print(len(word_dict))
    return word_dict

if __name__ == '__main__':

    rel_vocab = load_word_dictionary('../freebase_data/FB2M.rel.txt')
    torch.save(rel_vocab, 'vocab.rel.pt')

    ent_vocab = load_word_dictionary('../freebase_data/FB2M.ent.txt')
    torch.save(ent_vocab, 'vocab.ent.pt')

    rel1_vocab, rel2_vocab = load_rel_separated_dictionary('../freebase_data/FB2M.rel.txt')
    torch.save((rel1_vocab, rel2_vocab), 'vocab.rel.sep.pt')

    word_rel_vocab = creat_word_rel_dict('../freebase_data/FB2M.rel.pkl', '../data/QAData.test.pkl',
                                        '../data/QAData.train.pkl', '../data/QAData.valid.pkl')
    torch.save(word_rel_vocab, 'vocab.word&rel.pt')

