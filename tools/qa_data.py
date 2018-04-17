import numpy as np
from sklearn import preprocessing

def fb2www(in_data):
    if type(in_data) == type(' '):
        out_data = in_data.replace('.', '/').replace('fb:', 'www.freebase.com/')
    elif type(in_data) == type([]):
        out_data = [data.replace('.', '/').replace('fb:', 'www.freebase.com/') for data in in_data]
    return out_data

class QAData(object):
    """docstring for QAData"""
    def __init__(self, data_tuple):
        super(QAData, self).__init__()
        self.question = data_tuple[0]
        self.subject  = data_tuple[1]
        self.relation = data_tuple[2]
        self.object   = data_tuple[3]
        self.num_text_token = int(data_tuple[4])

    def add_candidate(self, sub, rels, types = None):
        if not hasattr(self, 'cand_sub'):
            self.cand_sub = []
        if not hasattr(self, 'cand_rel'):
            self.cand_rel = []
        if not hasattr(self, 'sub_rels'):
            self.sub_rels = []
        self.cand_sub.append(sub)
        self.sub_rels.append(rels)
        self.cand_rel.extend(rels)
        if types:
            if not hasattr(self, 'sub_types'):
                self.sub_types = []
            self.sub_types.append(types)

    def add_sub_types(self, types):
        if not hasattr(self, 'sub_types'):
            self.sub_types = []
        self.sub_types.append(types)

    def remove_duplicate(self):
        self.cand_rel = list(set(self.cand_rel))

    def make_score_mat(self):
        # make candidate unique rels
        self.num_sub  = len(self.cand_sub)
        self.num_rel  = len(self.cand_rel)
        self.rel_dict = {self.cand_rel[i]:i for i in range(self.num_rel)}
        
        # establish score matrix
        self.score_mat = np.zeros((self.num_sub, self.num_rel))
        for i in range(self.num_sub):
            for rel in self.sub_rels[i]:
                self.score_mat[i, self.rel_dict[rel]] = 1

    def fill_rel_score(self, scores):
        self.score_mat = self.score_mat * scores

    def fill_ent_score(self, scores):
        self.ent_score = preprocessing.scale(scores)

    def top_sub_rel(self):
        sub_score = np.sum(self.score_mat, 1)
        top_subscore = np.max(sub_score)
        top_subids = []
        for subid in np.argsort(sub_score)[::-1]:
            if sub_score[subid] < top_subscore:
                break
            top_subids.append(subid)

        top_relid = np.argmax(self.score_mat[top_subids[0]])

        return [self.cand_sub[subid] for subid in top_subids], self.cand_rel[top_relid]
