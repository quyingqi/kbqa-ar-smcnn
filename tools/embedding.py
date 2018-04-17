#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Mostly from https://github.com/OpenNMT/OpenNMT-py
# Most Code in load/save word2vec format refer to Gensim
import torch
import torch.nn as nn
from utils import load_word2vec_format, aeq

class Embeddings(nn.Module):

    def __init__(self,
                 word_vec_size,
                 dicts,
                 feat_merge='concat',
                 feat_vec_exponent=0.7,
                 feature_dicts=None,
                 feature_dims=None):
        """
        :param word_vec_size: Word Embedding Size
        :param dicts:         Word Dict
        :param feat_merge:    Merge action for the features embeddings.
        :param feat_vec_exponent:
                               When features embedding sizes are not set and using -feat_merge concat,
                               their dimension will be set to N^feat_vec_exponent where N is the number
                               of values the feature takes
        :param feature_dicts:
        """
        super(Embeddings, self).__init__()

        self.word_dict = dicts
        self.word_vec_size = word_vec_size
        self.feat_exp = feat_vec_exponent
        self.feat_merge = feat_merge

        # vocab_sizes: sequence of vocab sizes for words and each feature
        vocab_sizes = [self.word_dict.size()]

        # emb_sizes
        emb_sizes = [self.word_vec_size]
        if feature_dicts is not None and len(feature_dicts) > 0:
            vocab_sizes.extend(feat_dict.size() for feat_dict in feature_dicts)
            if self.feat_merge == 'concat':
                # Derive embedding sizes from each feature's vocab size
                emb_sizes.extend([int(feature_dim) for feature_dim in feature_dims])
            elif self.feat_merge == 'sum':
                # All embeddings to be summed must be the same size
                emb_sizes.extend(feature_dims)
            else:
                # TODO MLP
                raise NotImplementedError

        # Embedding Lookup Tables
        # [word_embedd, ...
        #  other embedding if has]
        self.emb_luts = nn.ModuleList([
            nn.Embedding(vocab, dim, padding_idx=self.word_dict.lookup(self.word_dict.pad_token))
            for vocab, dim in zip(vocab_sizes, emb_sizes)])

        self.init_model()

        self.output_size = self.embedding_size()

    def embedding_size(self):
        """
        Returns sum of all feature dimensions if the merge action is concat.
        Otherwise, returns word vector size.
        """
        if self.feat_merge == 'concat':
            return sum(emb_lut.embedding_dim
                       for emb_lut in self.emb_luts.children())
        else:
            return self.word_lookup_table.embedding_dim

    @property
    def word_lookup_table(self):
        return self.emb_luts[0]

    def init_model(self):
        for emb_table in self.emb_luts:
            emb_table.weight.data.normal_(0, 0.1)

    def load_pretrained_vectors(self, emb_file, binary=True, normalize=False):
        if emb_file is not None:
            pretrained, vec_size, vocab = load_word2vec_format(emb_file, self.word_dict.word2index,
                                                               binary=binary, normalize=normalize)

            # Init Out-of-PreTrain Wordembedding using Min,Max Uniform
            scale = torch.std(pretrained)
            # random_range = (torch.min(pretrained), torch.max(pretrained))
            random_range = (-scale, scale)
            random_init_count = 0
            for word in self.word_dict:

                if word not in vocab:
                    random_init_count += 1
                    nn.init.uniform(pretrained[self.word_dict.lookup(word)],
                                    random_range[0], random_range[1])

            self.word_lookup_table.weight.data.copy_(pretrained)
            print("Init %s words in uniform [%s, %s]" % (random_init_count, random_range[0], random_range[1]))
            return pretrained

    def merge(self, features):
        if self.feat_merge == 'concat':
            return torch.cat(features, 2)
        elif self.feat_merge == 'sum':
            return sum(features)
        else:
            return self.mlp(torch.cat(features, 2))

    def forward(self, inp):
        """
        Return the embeddings for words, and features if there are any.
        Args:
            inp (LongTensor): batch x len x nfeat
        Return:
            emb (Tensor): batch x len x self.embedding_size
        """
        if inp.dim() == 2:
            # batch x len
            emb = self.word_lookup_table(inp)
            return emb

        in_batch, in_length, nfeat = inp.size()
        aeq(nfeat, len(self.emb_luts))

        if len(self.emb_luts) == 1:
            emb = self.word_lookup_table(inp.squeeze(2))
        else:
            feat_inputs = (feat.squeeze(2)
                           for feat in inp.split(1, dim=2))
            features = [lut(feat)
                        for lut, feat in zip(self.emb_luts, feat_inputs)]
            emb = self.merge(features)

        out_batch, out_length, emb_size = emb.size()
        aeq(in_batch, out_batch)
        aeq(in_length, out_length)
        aeq(emb_size, self.embedding_size())

        return emb
