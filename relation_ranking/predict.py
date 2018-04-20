import os
import sys
import numpy as np
import torch
import pickle

from args import get_args
from seqRankingLoader import *
sys.path.append('../others')
sys.path.append('../tools')
import virtuoso

# please set the configuration in the file : args.py
args = get_args()
# set the random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("Warning: You have Cuda but do not use it. You are using CPU for training")


if not args.trained_model:
    print("ERROR: You need to provide a option 'trained_model' path to load the model.")
    sys.exit(1)

# load word vocab for questions, relation vocab for relations
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))
rel_vocab = torch.load(args.rel_vocab_file)
print('load relation vocab, size: %s' %len(rel_vocab))

os.makedirs(args.results_path, exist_ok=True)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

def evaluate(dataset = args.test_file, tp = 'test'):

    # load batch data for predict
    data_loader = SeqRankingLoader(dataset, args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d'
            %(tp, data_loader.batch_num, data_loader.batch_size))

    model.eval();
    n_correct = 0

    for data_batch_idx, data_batch in enumerate(data_loader.next_batch(shuffle=False)):
        pos_score1, pos_score2, pos_score3, neg_score1, neg_score2, neg_score3 = model(data_batch)
        neg_size, batch_size = pos_score1.size()
        n_correct += (torch.sum(torch.gt(pos_score1+pos_score2+pos_score3,
                                         neg_score1+neg_score2+neg_score3), 0).data ==
                      neg_size).sum()

    total = data_loader.batch_num*data_loader.batch_size
    accuracy = 100. * n_correct / (total)
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, n_correct, total))
    print("-" * 80)

def rel_pruned(neg_score, data):
    neg_rel = data.cand_rel
    pred_rel_scores = sorted(zip(neg_rel, neg_score), key=lambda i:i[1], reverse=True)
    pred_rel = pred_rel_scores[0][0]
    pred_sub = []
    for i, rels in enumerate(data.sub_rels):
        if pred_rel in rels:
            pred_sub.append((data.cand_sub[i], len(rels)))
    pred_sub = [sub[0] for sub in sorted(pred_sub, key = lambda sub:sub[1], reverse=True)]
    return pred_rel, pred_rel_scores, pred_sub


def predict(qa_pattern_file, tp):
    # load batch data for predict
    data_loader = CandidateRankingLoader(qa_pattern_file, word_vocab, rel_vocab, args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d' %(tp, data_loader.batch_num, 1))
    if args.write_res:
        results_file = open(os.path.join(args.results_path, '%s-pred_rel-wrong.txt' %tp), 'w')
        results_all_file = open(os.path.join(args.results_path, '%s-results-all.txt' %tp), 'w')

    model.eval()
    total = 0
    sub_correct = 0
    rel_scores = []
    n_correct = 0
    n_rel_correct = 0
    n_sub_recall = 0
    n_single_correct = 0
    for data_batch in data_loader.next_question():
        data = data_batch[-1]
        total += 1
        if data.subject not in data.cand_sub:
            continue
        sub_correct += 1

        pos_score, neg_score = model(data_batch[:-1])
        neg_score = neg_score.data.squeeze().cpu().numpy()

        if args.write_score:
            rel_scores.append((data.cand_rel, data.relation, neg_score))

        pred_rel, pred_rel_scores, pred_sub = rel_pruned(neg_score, data)

        if pred_rel == data.relation:
            n_rel_correct += 1
            if data.subject in pred_sub:
                n_sub_recall += 1
                if pred_sub[0] == data.subject:
                    n_correct += 1
                    if len(pred_sub) == 1:
                        n_single_correct += 1

    if args.write_score:
        score_file = open(os.path.join(args.results_path, 'score-rel-%s.pkl' %tp), 'wb')
        pickle.dump(rel_scores, score_file)

    accuracy = 100. * n_correct / total
    rel_acc = 100. * n_rel_correct / sub_correct
    sub_recall = 100. * n_sub_recall / sub_correct
    single_acc = 100. * n_single_correct / sub_correct
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, n_correct, total))
    print('rel_acc: ', rel_acc, n_rel_correct, sub_correct)
    print('recall: ', sub_recall, n_sub_recall, sub_correct)
    print('single_acc: ', single_acc, n_single_correct, sub_correct)
    print("-" * 80)

if args.predict:
    qa_pattern_file = '../entity_detection/results/QAData.label.%s.pkl'
    for tp in ('valid', 'test', 'train'):
        predict(qa_pattern_file % tp, tp)
else:
    evaluate(args.valid_file, "valid")
    evaluate(args.test_file, "test")
    evaluate(args.train_file, 'train')
