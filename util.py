import sys
import copy
import random
import numpy as np
from collections import defaultdict
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plot
from sampler import Dataset

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def random_neq_from_session(os, itemnum, ts):
    diff = os.difference(ts)
    if bool(diff): #not empty
        t = random.sample(diff, 1)
        return t[0]
    #else if empty    
    t = np.random.randint(1, itemnum+1)

    while t in ts:
        t = np.random.randint(1, itemnum+1)
    return t

def data_partition(fname, percentage=[0.1, 0.2]):
    itemnum = 0

    sessions = defaultdict(list)
    session_train = []
    session_valid = []
    session_test = []
    # assume user/item index starting from 1
    session_id = 0
    f = open('data/%s.csv' % fname, 'r')
    total_length = 0
    max_length = 0
    for line in f:

        items = [int(l) for l in line.rstrip().split(',')]

        if len(items) < 5: continue
        total_length += len(items)

        if max_length< len(items):
            max_length = len(items)
        
        itemnum = max(max(items), itemnum)
        sessions[session_id].append(items)
        session_id += 1

    print("Avg length:", total_length/session_id)
    print("Maximum length:", max_length)

    valid_perc = percentage[0]
    test_perc = percentage[1]

    total_sessions = session_id
    
    #np.random.seed(10)

    shuffle_indices = np.random.permutation(range(total_sessions)) #
    
    train_index = int(total_sessions*(1 - valid_perc - test_perc))
    valid_index = int(total_sessions*(1 - test_perc))

    if (train_index == valid_index): valid_index += 1 #break the tie
    
    train_indices = shuffle_indices[:train_index]
    valid_indices = shuffle_indices[train_index:valid_index]
    test_indices = shuffle_indices[valid_index:]


    for i in train_indices:
        session_train.extend(sessions[i])
    for i in valid_indices:
        session_valid.extend(sessions[i])
    for i in test_indices:
        session_test.extend(sessions[i])
    
    
    return [np.asarray(session_train), np.asarray(session_valid), np.asarray(session_test), itemnum]


def saveAsNextItNetFormat(fname, maxlen):
    import csv
    import numpy as np
        
    sessions = []

    # assume user/item index starting from 1
    f = open('data/%s.csv' % fname, 'r')

    for line in f:

        items = [int(l) for l in line.rstrip().split(',')]

        if len(items) < 5: continue
        
        seq = np.zeros([maxlen], dtype=np.int32)
        
        idx = maxlen - 1

        for i in reversed(items):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        
        sessions.append(seq)
        
    print("number of session:", len(sessions))

    with open(fname+'_nextitnet_format.csv',"w",newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(sessions)



def saveAsGRUFormat(fname, user_train, user_valid, user_test):
    
    session_id = 0
    train = []
    for session in user_train:
        for item in session:
            train.append([session_id, item, 0])
        session_id += 1

    valid = []
    for session in user_valid:
        for item in session:
            valid.append([session_id, item, 0])
        session_id += 1

    test = []
    for session in user_test:
        for item in session:
            test.append([session_id, item, 0])
        session_id += 1

    train_data = pd.DataFrame(train, columns= ['SessionId', 'ItemId', 'Time'])
    valid_data = pd.DataFrame(valid, columns= ['SessionId', 'ItemId', 'Time'])
    test_data = pd.DataFrame(test, columns= ['SessionId', 'ItemId', 'Time'])

    train_data.to_csv(fname+'_grurec_train_data.csv',  sep=' ', index=None)
    valid_data.to_csv(fname+'_grurec_valid_data.csv',  sep=' ', index=None)
    test_data.to_csv(fname+'_grurec_test_data.csv',  sep=' ', index=None)


def evaluate(model, test_sessions, itemnum, args, computing_device, unbiased_estimation=False):
    #set the environment
    model.eval()
    
    MRR = 0.0
    NDCG = 0.0
    HT = 0.0

    MRR_plus_10 = 0.0
    NDCG_plus_10 = 0.0
    HT_plus_10 = 0.0

    valid_sessions = 0.0

    #putting the item in the top

    all_items = np.array(range(1, itemnum+1))
    all_items_tensor = torch.LongTensor(all_items).to(computing_device, non_blocking=True)

    dataset = Dataset(test_sessions, args, itemnum, False)

    sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    with torch.no_grad():
            
        for step, (seq, grouth_truth) in tqdm(enumerate(sampler), total=len(sampler)): 

            #safety check
            
            seq = torch.LongTensor(seq).to(computing_device, non_blocking=True)
            
            _, rank_20 = model.forward(seq, test_item = all_items_tensor)

            rank_20 = rank_20.cpu().detach().numpy()
            grouth_truth = grouth_truth.view(-1, 1).numpy()
            

            try:
                ranks = np.where(rank_20 == grouth_truth)

                try:
                    ranks = ranks[1]
                except:
                    ranks = ranks[0]

                for rank in ranks:

                    if rank < args.top_k:
                        
                        MRR += 1.0/(rank + 1)
                        NDCG += 1 / np.log2(rank + 2)
                        HT += 1

                    if rank < args.top_k + 10:

                        MRR_plus_10 += 1.0/(rank + 1)
                        NDCG_plus_10 += 1 / np.log2(rank + 2)
                        HT_plus_10 += 1
                
            except:
                continue
                    
        valid_sessions = len(test_sessions)

    return MRR / valid_sessions, NDCG / valid_sessions, HT / valid_sessions, MRR_plus_10 / valid_sessions, NDCG_plus_10 / valid_sessions, HT_plus_10 / valid_sessions

