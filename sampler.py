import numpy as np

import torch
from torch.utils import data

class Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, data, args, itemnum, train):
            'Initialization'
            self.data = data
            self.args = args
            self.itemnum = itemnum
            self.train = train

    def __len__(self):
            'Denotes the total number of samples'            
            return len(self.data)

    def __train__(self, index):
            
            session = np.asarray(self.data[index], dtype=np.int64)
    
            if len(session) > self.args.maxlen:
                session = session[-self.args.maxlen:]
            else:
                session = np.pad(session, (self.args.maxlen-len(session), 0), 'constant', constant_values=0)

            curr_seq = session[:-1]
            curr_pos = session[1:]

            return curr_seq, curr_pos
    
    def __test__(self, index):

            session = self.data[index]

            seq = np.zeros([self.args.maxlen], dtype=np.int64)
            idx = self.args.maxlen - 1

            for i in reversed(session[:-1]): #everything except the last one
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            return seq, session[-1]-1 #Index # np.where(self.all_items == session[-1])[0][0]

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample

            if self.train:
                return self.__train__(index)
            else:
                return self.__test__(index)
            


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t