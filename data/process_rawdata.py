# This tool preprocess the original simple question dataset in 5 aspects:
#   1. change triple information in to fb:... format 
#   2. replace the escape ('//') simbol in original question
#   3. tokenize the question
#   4. change the tokenized question into lower cases
#   5. add another fields which indicates the token number of the question
#   6. All pairs are sorted by the length of the question, and stored to QAData.*.pkl

import multiprocessing as mp
import sys, os, io, re
import pickle
from nltk import word_tokenize
sys.path.append('../tools')
from qa_data import QAData
import virtuoso

split = None

def extract(line):
    fields   = line.strip().split('\t')
    sub = 'fb:' + fields[0].split('www.freebase.com/')[-1].replace('/','.')
    rel = 'fb:' + fields[1].split('www.freebase.com/')[-1].replace('/','.')
    obj = 'fb:' + fields[2].split('www.freebase.com/')[-1].replace('/','.')
    if sub == 'fb:m.07s9rl0':
        sub = 'fb:m.02822'
    if obj == 'fb:m.07s9rl0':
        obj = 'fb:m.02822'
    question = fields[-1].replace('\\\\','')
    tokens   = word_tokenize(question)
    question = ' '.join(tokens).lower()
    tmp = question.split(' . ')
    return '. '.join(tmp), sub, rel, obj, len(tokens)-len(tmp)+1

def get_indices(src_list, pattern_list):
    indices = None
    for i in range(len(src_list)):
        match = 1
        for j in range(len(pattern_list)):
            if src_list[i+j] != pattern_list[j]:
                match = 0
                break
        if match:
            indices = range(i, i + len(pattern_list))
            break
    return indices

def query_golden_subs(data):
    golden_subs = []
    if data.text_subject:
        # extract fields needed
        relation     = data.relation
        subject      = data.subject
        text_subject = data.text_subject
        
        # query name / alias by subject (id)
        candi_sub_list = virtuoso.str_query_id(text_subject)

        # add candidates to data
        for candi_sub in candi_sub_list:
            candi_rel_list = virtuoso.id_query_out_rel(candi_sub)
            if relation in candi_rel_list:
                golden_subs.append(candi_sub)

    if len(golden_subs) == 0:
        golden_subs = [data.subject]

    return golden_subs

def search_ngrams():
    pass

def reverse_link(question, subject):
    # get question tokens
    tokens = question.split()

    # init default value of returned variables
    text_subject = None
    text_attention_indices = None

    # query name / alias by node_id (subject)
    res_list = virtuoso.id_query_str(subject)

    # sorted by length
    for res in sorted(res_list, key = lambda res: len(res), reverse = True):
        pattern = r'(^|\s)(%s)($|\s)' % (re.escape(res))
        if re.search(pattern, question):
            text_subject = res
            text_attention_indices = get_indices(tokens, res.split())
            break

    return text_subject, text_attention_indices

def form_question_pattern(data):
    question_pattern = None
    if data.text_attention_indices:
        anonymous_tokens = []
        tokens = data.question.split()
        anonymous_tokens.extend(tokens[:data.text_attention_indices[0]])
        anonymous_tokens.append('X')
        anonymous_tokens.extend(tokens[data.text_attention_indices[-1]+1:])
        question_pattern = ' '.join(anonymous_tokens)

    return question_pattern

def knowledge_graph_attributes(data_list, pid = 0):
    # Open log file
    log_file = open('logs/log.%s.%d.txt'%(split, pid), 'w')
    log_file.write('total length: %d\n' %(len(data_list)))

    succ_att_link = 0
    qadata_list = []
    for data_index, data_tuple in enumerate(data_list):
        # Step-1: create QAData instance
        data = QAData(data_tuple)

        # Step-2: reverse linking
        data.text_subject, data.text_attention_indices = reverse_link(data.question, data.subject)

        # Step-3: create question pattern
        data.question_pattern = form_question_pattern(data)

        qadata_list.append(data)
        
        # logging
        if data.text_subject:
            succ_att_link += 1
#        log_file.write('[%d] attention: %f\n' % (data_index, succ_att_link / float(data_index+1)))
        log_file.write('[%d] %s\t%s\t%s\n' %(data_index, data.question, data.subject, data.text_subject))

    pickle.dump(qadata_list, open('temp.%s.pkl'%(pid), 'wb'))
    log_file.write('total: %d\tattack: %d\trate: %f\n' % (len(data_list), succ_att_link,
                                                          succ_att_link/len(data_list)))
    log_file.close()

def process(num_process, data_list):
    # Make dir
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Split workload
    length = len(data_list)
    data_per_p = (length + num_process - 1) // num_process

    # Spawn processes
    processes = [
        mp.Process(
            target = knowledge_graph_attributes,
            args = ( 
                data_list[pid*data_per_p:(pid+1)*data_per_p],
                pid
                )
            )
        for pid in range(num_process)
    ]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print('python preprocess.py input_file num_process')
        sys.exit(-1)

    in_file_path = sys.argv[1]
    num_process = int(sys.argv[2])

    split = in_file_path.split('_')[-1].split('.')[0]

    in_file = io.open(in_file_path, 'r', encoding='utf8')
    
    data_list = []
    for line in in_file:
        question, sub, rel, obj, length = extract(line)
        data_list.append((question, sub, rel, obj, length))

    process(num_process, sorted(data_list, key = lambda data: data[-1], reverse = True))

    # Merge all data [this will preserve the order]
    new_data_list = []
    for p in range(num_process):
        temp_fn = 'temp.%d.pkl'%(p)
        new_data_list.extend(pickle.load(open(temp_fn, 'rb')))
        os.remove(temp_fn)

    pickle.dump(new_data_list, open('QAData.%s.pkl'%(split), 'wb'))

    in_file.close()
