
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import *





class ConvRec(nn.Module):
    
    def __init__(self, args, itemnum):
        super(ConvRec, self).__init__()
        
        add_args(args)

        self.args = args
        self.dropout = args.dropout
        self.maxlen = args.maxlen
        self.itemnum = itemnum

        self.item_embedding = Embedding(itemnum + 1, args.embed_dim, 0)
        self.embed_scale = math.sqrt(args.embed_dim)  
        self.position_encoding = Embedding(args.maxlen, args.embed_dim, 0)
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            ConvRecLayer(args, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.layers)
        ])
        
        self.layer_norm = LayerNorm(args.embed_dim)

    def forward(self, seq, pos=None, neg=None, test_item = None):
        
        # positions = self.position_encoding(torch.unsqueeze(torch.arange(seq.size(1)), 0).repeat(seq.size(0), 1).to(self.args.computing_device))#embed
        
        # mask = torch.unsqueeze(torch.ne(seq, 0).type(torch.FloatTensor), -1).to(self.args.computing_device)
        # x = self.embed_scale * self.item_embedding(seq)
        x =  self.item_embedding(seq)
        

        # x += positions
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x *= mask

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x = self.layer_norm(x)
            x, attn = layer(x)
            inner_states.append(x)

        # if self.normalize:
        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        
        seq_emb = x.contiguous().view(-1, x.size(-1)) # reshaping it to [tf.shape(self.input_seq)[0] * args.maxlen x args.hidden_units
        pos_logits = None 
        neg_logits = None 
        rank_20 = None 
        istarget = None
        loss = None

        if pos is not None:
            pos = torch.reshape(pos, (-1,))
            #batch x num_neg x seq_leng -> num_neg x batch x seq_leng
            # neg = neg.transpose(0, 1)
            # neg = torch.reshape(neg, (neg.size(0), -1,)) #TODO
            
            # pos_emb = self.user_item_embedding(pos) #can just be replaced by model forward
            # neg_emb = self.user_item_embedding(neg)
            
            # # prediction layer
            # #MF
            # pos_logits = torch.sum(pos_emb * user_emb, -1)
            # neg_logits = torch.sum(neg_emb * user_emb, -1)
            nnz = torch.ne(pos, 0).nonzero().squeeze(-1)#.type(torch.FloatTensor)
            neg = torch.randint(1,self.itemnum+1, (self.args.num_neg_samples, nnz.size(0)), device=self.args.computing_device) #TODO

            pos_emb = self.item_embedding(pos[nnz]) #can just be replaced by model forward
            neg_emb = self.item_embedding(neg)
            seq_emb = seq_emb[nnz]

            #sequential context
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            # neg_logits = torch.einsum('nbe, be->nb', neg_emb, seq_emb)

            # # neg_logits = torch.sum(neg_emb * seq_emb, -1)
            # #more than one negatives
            # # num of negs x batch * leng x embed_size
            # stacked_seq_em = seq_emb.unsqueeze(0).repeat(neg.size(0), 1, 1)
            # # num of negs * batch * leng x 1 x embed_size
            # stacked_seq_em = stacked_seq_em.contiguous().view(-1, 1, x.size(-1))
            # # num of negs * batch * leng x embed_size x 1
            # neg_emb = neg_emb.contiguous().view(-1, x.size(-1), 1)
            # # num of negs * batch * leng  x 1
            # neg_logits = torch.bmm(stacked_seq_em, neg_emb).view(neg.size(0), seq_emb.size(0)) #check
            negative_scores = torch.sum((1 - torch.sigmoid(neg_logits) + 1e-24).log(), axis = 0)

            loss = torch.sum(-(torch.sigmoid(pos_logits) + 1e-24).log() - negative_scores)/nnz.size(0)

            # test_logits = test_logits.contiguous().view(seq.size(0), self.maxlen, neg_item.size(0))
            # test_logits = test_logits[:, -1, :]
            # negative_scores = (1 - torch.sigmoid(neg_logits[0]) + 1e-24).log() * istarget
            # for i in range(1, self.args.num_neg_samples):
            #     negative_scores += (1 - torch.sigmoid(neg_logits[i]) + 1e-24).log() * istarget
            
            
                
        if test_item is not None:            
            #self.test_item = tf.placeholder(tf.int32, shape=(101))
            # test_item_emb = self.user_item_embedding(test_item)
            # test_logits = torch.mm(user_emb, test_item_emb.t()) #check

            test_item_emb = self.item_embedding(test_item)
            seq_emb = seq_emb.view(seq.size(0), seq.size(1), -1)
            seq_emb = seq_emb[:, -1, :]
            seq_emb = seq_emb.contiguous().view(-1, seq_emb.size(-1))
            test_logits = torch.mm(seq_emb, test_item_emb.t()) #check

            test_logits_indices = torch.argsort(-test_logits)
            rank_20 = test_logits_indices[:, :20]
            # test_logits = test_logits.contiguous().view(seq.size(0), 1, test_item.size(0))
            # test_logits = test_logits[:, -1, :]

        return loss, rank_20

    
class ConvRecLayer(nn.Module):
   

    def __init__(self, args,  kernel_size=0):
        super().__init__()
        self.embed_dim = args.embed_dim
        
        self.conv = DynamicConv1dTBC(args.embed_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=args.weight_softmax,
                                         num_heads=args.heads,
                                         unfold = None, #TODO: unfold is a trick introduced in original paper, omitting it now
                                         weight_dropout=args.weight_dropout)
        
        self.dropout = args.dropout
        self.layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.ffn_embed_dim)
        self.fc2 = Linear(args.ffn_embed_dim, self.embed_dim)


    def forward(self, x, conv_mask=None,
                conv_padding_mask=None):
        
        T, B, C = x.size()
        
        x = self.conv(x)
        x = self.layer_norm(x)  
        attn = None
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        return x, attn


def add_args(args):
    #some additional arguments

    if len(args.decoder_kernel_size_list) == 1:
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.layers

    args.weight_softmax = True

    print("model arguments", args)