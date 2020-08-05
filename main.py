#%%
import os
import time
import argparse
import csv
#%%
import numpy as np
#%%
from sampler import Dataset
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
    parser.add_argument('--maxlen', default=30, type=int)

    parser.add_argument('--embed_dim', default=200, type=int) 
    parser.add_argument('--ffn_embed_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.2, type=float) #Helps to converge faster
    parser.add_argument('--weight_dropout', default=0.2, type=float)


    parser.add_argument('--layers', default=2, type=int) 
    parser.add_argument('--heads', default=200, type=int) 

    parser.add_argument('--decoder_kernel_size_list', default = [5, 5]) #depends on the number of layer
    
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_neg_samples', default = 400, type=int) #TODO 100 is sufficient
    parser.add_argument('--eval_epoch', default = 5, type=int)
    
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
        action += np.count_nonzero(i)
    
    for i in valid:
        action += np.count_nonzero(i)
    
    
    for i in test:
        action += np.count_nonzero(i)

    print("number of actions,", action)
    
    print("average length of sessions,", action/(len(train)+len(valid)+len(test)))


    # TODO: for running other baselines
    # saveAsNextItNetFormat(args.dataset, args.maxlen)
    # saveAsGRUFormat(args.dataset, train, valid, test)

    num_batch = len(train) // args.batch_size
    print(num_batch)
    #%%
    
    f = open(os.path.join(result_path, 'log.txt'), 'w')
    
    conv_model = ConvRec(args, itemnum)
    conv_model = conv_model.to(computing_device, non_blocking=True)
    
    # TODO: testing a pretrained model
    # if os.path.exists(result_path+"pretrained_model.pth"):
    #     conv_model = ConvRec(args, itemnum)#, True)
    #     conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))
    #     conv_model = conv_model.to(computing_device)        
    #     t_test = evaluate(conv_model, test, itemnum, args, computing_device, False)
    #     model_performance = "model performance on test"+str(t_test)
    #     print (model_performance)


    T = 0.0
    t0 = time.time()


    optimizer = optim.Adam(conv_model.parameters(), lr = args.lr, betas=(0.9, 0.98),weight_decay = 0.0)

    f.write(str(args)+'\n')
    f.flush()

    # loss_log = open(os.path.join(opts.checkpoints_dir, 'loss_log.txt'), 'w')
    

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    best_ndcg = 0
    best_hit = 0
    model_performance = None

    stop_count = 0
    total_epochs = 1    
    
    dataset = Dataset(train, args, itemnum, train=True)

     
    sampler = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    for epoch in range(1, args.num_epochs + 1):  
        conv_model.train()

        epoch_losses = []

        
        for step, (seq, pos) in tqdm(enumerate(sampler), total=len(sampler)):  
                
            optimizer.zero_grad()


            seq = torch.LongTensor(seq).to(computing_device, non_blocking=True)
            pos = torch.LongTensor(pos).to(computing_device, non_blocking=True)
            # neg = torch.LongTensor(neg).to(computing_device, non_blocking=True)
            
            # pos = torch.reshape(pos, (-1,))
            # #batch x num_neg x seq_leng -> num_neg x batch x seq_leng
            # neg = neg.transpose(0, 1)
            # neg = torch.reshape(neg, (neg.size(0), -1,)) #TODO

            # print("asi")
            loss, _  = conv_model.forward(seq, pos=pos)#, neg=neg)

            
            # istarget = torch.ne(pos, 0).type(torch.FloatTensor).to(computing_device, non_blocking=True)

            
            # #TODO - improve the folllowing as we are comparing against all items, we have to push more for ranking
            # negative_scores = torch.zeros(logits['neg_logits'].size(1)).to(computing_device, non_blocking=True)
            # #
            # for i in range(args.num_neg_samples):
            #     negative_scores += (1 - torch.sigmoid(logits['neg_logits'][i]) + 1e-24).log() * istarget

            # loss = torch.sum(-(torch.sigmoid(logits['pos_logits']) + 1e-24).log() * istarget - logits['neg_logits'])/torch.sum(istarget)


            epoch_losses.append(loss.item())

            # Compute gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            
        
        if total_epochs % args.eval_epoch == 0:

            t_valid = evaluate(conv_model, valid, itemnum, args, computing_device)
            
            print ('\nnum of steps:%d, time: %f(s), valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f), valid (MRR@%d: %.4f, NDCG@%d: %.4f, HR@%d: %.4f)' % (total_epochs, T, args.top_k, t_valid[0], args.top_k, t_valid[1], args.top_k, t_valid[2],
            args.top_k+10, t_valid[3], args.top_k+10, t_valid[4], args.top_k+10, t_valid[5]))

            f.write(str(t_valid) + '\n')
            f.flush()

            
            if t_valid[0]>best_ndcg:
                best_ndcg = t_valid[0]
                torch.save(conv_model.state_dict(), result_path+"pretrained_model.pth")
                stop_count = 1
            else:
                stop_count += 1

            if stop_count == 3: #model did not improve 3 consequetive timesprint ('Evaluating'),
                break
                
        total_epochs += 1

        train_loss = np.mean(epoch_losses)
        print(str(epoch) + "epoch loss", train_loss)
        
    
        
    conv_model = ConvRec(args, itemnum)
    conv_model.load_state_dict(torch.load(result_path+"pretrained_model.pth"))
    
    conv_model = conv_model.to(computing_device)
        
    t_test = evaluate(conv_model, test, itemnum, args, computing_device)

    model_performance = "model performance on test"+str(t_test)
    print(model_performance)

    f.write(model_performance+'\n')
    f.flush()
    f.close()

    print("Done")


# %%
