import os
import sys
import numpy as np
import torch
import pickle

from args import get_args
from model import EntityDetection
from evaluation import evaluation
from seqLabelingLoader import SeqLabelingLoader
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

# load word vocab for questions
word_vocab = torch.load(args.vocab_file)
print('load word vocab, size: %s' % len(word_vocab))

os.makedirs(args.results_path, exist_ok=True)

# load the model
model = torch.load(args.trained_model, map_location=lambda storage,location: storage.cuda(args.gpu))

def predict(dataset=args.test_file, tp='test', save_qadata=args.save_qadata):
    # load QAdata
    qa_data_path = '../data/QAData.%s.pkl' % tp
    qa_data = pickle.load(open(qa_data_path,'rb'))

    # load batch data for predict
    data_loader = SeqLabelingLoader(dataset, args.gpu)
    print('load %s data, batch_num: %d\tbatch_size: %d'
            %(tp, data_loader.batch_num, data_loader.batch_size))

    model.eval();

    n_correct = 0
    n_correct_sub = 0
    n_correct_extend = 0
    n_empty = 0
    linenum = 1
    qa_data_idx = 0

    new_qa_data = []

    gold_list = []
    pred_list = []

    for data_batch_idx, data_batch in enumerate(data_loader.next_batch(shuffle=False)):
        if data_batch_idx % 50 == 0:
            print(tp, data_batch_idx)
        scores = model(data_batch)
        n_correct += ((torch.max(scores, 1)[1].view(data_batch[1].size()).data ==
                            data_batch[1].data).sum(dim=0) == data_batch[1].size()[0]).sum()

        index_tag = np.transpose(torch.max(scores, 1)[1].view(data_batch[1].size()).cpu().data.numpy())
        gold_tag = np.transpose(data_batch[1].cpu().data.numpy())
        index_question = np.transpose(data_batch[0].cpu().data.numpy())

        gold_list.append(np.transpose(data_batch[1].cpu().data.numpy()))
        pred_list.append(index_tag)

        for i in range(data_loader.batch_size):
            while qa_data_idx < len(qa_data) and not qa_data[qa_data_idx].text_subject:
                qa_data_idx += 1
            if qa_data_idx >= len(qa_data):
                break
            _qa_data = qa_data[qa_data_idx]
            tokens = np.array(_qa_data.question.split())
            pred_text = ' '.join(tokens[np.where(index_tag[i][:len(tokens)])])

            pred_sub, pred_sub_extend = get_candidate_sub(tokens, index_tag[i])
            if _qa_data.subject in pred_sub:
                n_correct_sub += 1
            if _qa_data.subject in pred_sub_extend:
                n_correct_extend += 1
            if not pred_sub_extend:
                n_empty += 1

            if save_qadata:
                for sub in pred_sub_extend:
                    rel = virtuoso.id_query_out_rel(sub)
                    _qa_data.add_candidate(sub, rel)
                if hasattr(_qa_data, 'cand_rel'):
                    _qa_data.remove_duplicate()

#                if _qa_data.subject not in pred_sub_extend:
#                    _qa_data.neg_rel = virtuoso.id_query_out_rel(_qa_data.subject)

                new_qa_data.append((_qa_data, len(_qa_data.question_pattern.split())))

            linenum += 1
            qa_data_idx += 1

    total = linenum-1
    accuracy = 100. * n_correct / total
    print("%s\taccuracy: %8.6f\tcorrect: %d\ttotal: %d" %(tp, accuracy, n_correct, total))
    P, R, F = evaluation(gold_list, pred_list)
    print("Precision: {:10.6f}% Recall: {:10.6f}% F1 Score: {:10.6f}%".format(100. * P, 100. * R, 100. * F))

    sub_accuracy = 100. * n_correct_sub / total
    print('subject accuracy: %8.6f\tcorrect: %d\ttotal:%d' %(sub_accuracy, n_correct_sub, total))

    extend_accuracy = 100. * n_correct_extend / total
    print('extend accuracy: %8.6f\tcorrect: %d\ttotal:%d' %(extend_accuracy, n_correct_extend, total))

    print('suject not found: %8.6f\t%d' %(n_empty/total, n_empty))
    print("-" * 80)

    if save_qadata:
        qadata_save_path = open(os.path.join(args.results_path, 'QAData.label.%s.pkl' %(tp)), 'wb')
        data_list = [data[0] for data in sorted(new_qa_data, key = lambda data: data[1],
                                                reverse=True)]
        pickle.dump(data_list, qadata_save_path)

def get_candidate_sub(question_tokens, pred_tag):
    flag = False
    starts = []
    ends = []
    for i, tag in enumerate(pred_tag):
        if tag==1 and not flag:
            starts.append(i)
            flag = True
        elif tag==0 and flag:
            if (i+1 < len(question_tokens) and pred_tag[i+1]==0) or i+1==len(question_tokens):
                ends.append(i-1)
                flag = False
    if flag:
        ends.append(len(question_tokens)-1)

    sub_list = []
    shift = [0,-1,1,-2,2]
    pred_sub = []
    for left in shift:
        for right in shift:
            for i in range(len(starts)):
                if starts[i]+left < 0:continue
                if ends[i]+1+right > len(question_tokens):continue
                text = question_tokens[starts[i]+left:ends[i]+1+right]
                subject = virtuoso.str_query_id(' '.join(text))
#                print(text, subject)
                if left==0 and right==0:
                    pred_sub = subject
                sub_list.extend(subject)
            if sub_list:
                return pred_sub, sub_list
    return pred_sub, sub_list

# run the model on the dev set and write the output to a file
predict(args.valid_file, "valid")

# run the model on the test set and write the output to a file
predict(args.test_file, "test")

# run the model on the train set and write the output to a file
predict(args.train_file, 'train')
