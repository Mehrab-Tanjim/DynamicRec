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
        
 
        x =  self.item_embedding(seq)
        

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
        
        seq_emb = x.contiguous().view(-1, x.size(-1)) # reshaping it to [arg.batch_size x args.maxlen * args.hidden_units]
        pos_logits = None 
        neg_logits = None 
        rank_20 = None 
        istarget = None
        loss = None

        if pos is not None:
            pos = torch.reshape(pos, (-1,))

            nnz = torch.ne(pos, 0).nonzero().squeeze(-1)
            neg = torch.randint(1,self.itemnum+1, (self.args.num_neg_samples, nnz.size(0)), device=self.args.computing_device)

            pos_emb = self.item_embedding(pos[nnz])
            neg_emb = self.item_embedding(neg)
            seq_emb = seq_emb[nnz]

            #sequential context
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)
            negative_scores = torch.sum((1 - torch.sigmoid(neg_logits) + 1e-24).log(), axis = 0)

            loss = torch.sum(-(torch.sigmoid(pos_logits) + 1e-24).log() - negative_scores)/nnz.size(0)

                
        if test_item is not None:        

            test_item_emb = self.item_embedding(test_item)
            seq_emb = seq_emb.view(seq.size(0), seq.size(1), -1)
            seq_emb = seq_emb[:, -1, :]
            seq_emb = seq_emb.contiguous().view(-1, seq_emb.size(-1))
            test_logits = torch.mm(seq_emb, test_item_emb.t()) #check

            test_logits_indices = torch.argsort(-test_logits)
            rank_20 = test_logits_indices[:, :20]

        return loss, rank_20

    
class ConvRecLayer(nn.Module):
   

    def __init__(self, args,  kernel_size=0):
        super().__init__()
        self.embed_dim = args.embed_dim
        
        self.conv = DynamicConv1dTBC(args.embed_dim, kernel_size, padding_l=kernel_size-1,
                                         weight_softmax=args.weight_softmax,
                                         num_heads=args.heads,
                                         unfold = None, 
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

    if len(args.decoder_kernel_size_list) == 1: # For safety in case kernel size list does not match with # of convolution layers
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.layers

    args.weight_softmax = True

    print("Model arguments", args)