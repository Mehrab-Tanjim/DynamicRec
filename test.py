import os
import time
import argparse
import csv
#%%
import numpy as np
from sampler import WarpSampler
from model import ConvRec
from tqdm import tqdm
# from util import *
import pickle
from util_session import *
#%%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#%%



def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #--dataset=Video --train_dir=default 
    parser.add_argument('--dataset', default='nowplaying')
    parser.add_argument('--top_k', default=10, type=int)
    
    parser.add_argument('--train_dir', default='dev')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)

    #model specific -- may need to change in model.py
    parser.add_argument('--maxlen', default=20, type=int)

    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--ffn_embed_dim', default=100, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--weight_dropout', default=0.0, type=float)


    parser.add_argument('--layers', default=2, type=int) 
    parser.add_argument('--heads', default=1, type=int) 

    parser.add_argument('--decoder_kernel_size_list', default = [5, 5]) #depends on the number of layer
    
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_neg_samples', default = 100, type=int)
    parser.add_argument('--eval_step', default = 5000, type=int)
    
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    
    parser.add_argument('--computing_device', default=computing_device)

    # # Get the arguments
    try:
        #if running from command line
        args = parser.parse_args()
    except:
        #if running in ides
        args = parser.parse_known_args()[0] 

    
    result_path = 'results/'+args.dataset + '_' + args.train_dir

    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    
    if os.path.exists("data/"+args.dataset + '.pkl'):
        pickle_in = open("data/"+args.dataset+".pkl","rb")
        dataset = pickle.load(pickle_in)
    else:
        dataset = data_partition(args.dataset)
        pickle_out = open("data/"+args.dataset+".pkl","wb")
        pickle.dump(dataset, pickle_out)
        pickle_out.close()
    
    
    [train, valid, test, itemnum] = dataset
    
    print("number of sessions,",len(train)+len(valid)+len(test))
    print("number of items,", itemnum)

    action = 0
    for i in train:
        action += len(i)
    
    for i in valid:
        action += len(i)
    
    
    for i in test:
        action += len(i)

    print("number of actions,", action)
    
    print("average length of sessions,", action/(len(train)+len(valid)+len(test)))


    # TODO: for running other baselines
    # saveAsNextItNetFormat(args.dataset, args.maxlen)
    # saveAsGRUFormat(args.dataset, train, valid, test)

    num_batch = len(train) // args.batch_size
    print(num_batch)
    #%%
    
    f = open(os.path.join(result_path, 'log.txt'), 'w')
    
    conv_model = ConvRec(args, itemnum)#, True)
    conv_model = conv_model.to(computing_device)

    # TODO: testing a pretrained model
    if os.path.exists(result_path+"pretrained_model.pth"):
        conv_model = ConvRec(args, itemnum)#, True)
        conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))
        conv_model = conv_model.to(computing_device)        
        t_test = evaluate(conv_model, test, itemnum, args, computing_device, f)
        model_performance = "model performance on test"+str(t_test)
        print (model_performance)
