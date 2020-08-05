
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def unfold1d(x, kernel_size, padding_l, pad_value=0):
    '''unfold T x B x C to T x B x C x K'''
    if kernel_size > 1:
        T, B, C = x.size()
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B*C, C, 1, B*C))
    else:
        x = x.unsqueeze(3)
    return x

class DynamicConv1dTBC(nn.Module):
    '''Dynamic lightweight convolution taking T x B x C inputs
    Args:
        input_size: # of channels of the input
        kernel_size: convolution channels
        padding_l: padding to the left when using "same" padding
        num_heads: number of heads used. The weight is of shape (num_heads, 1, kernel_size)
        weight_dropout: the drop rate of the DropConnect to drop the weight
        weight_softmax: normalize the weight with softmax before the convolution
        renorm_padding: re-normalize the filters to ignore the padded part (only the non-padding parts sum up to 1)
        bias: use bias
        conv_bias: bias of the convolution
        query_size: specified when feeding a different input as the query
        in_proj: project the input and generate the filter together

    Shape:
        Input: TxBxC, i.e. (timesteps, batch_size, input_size)
        Output: TxBxC, i.e. (timesteps, batch_size, input_size)

    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias:   the learnable bias of the module of shape `(input_size)`
    '''
    def __init__(self, input_size, kernel_size=1, padding_l=None, num_heads=1, unfold=False,
                 weight_dropout=0., weight_softmax=False,
                 renorm_padding=False, bias=False, conv_bias=False,
                 query_size=None):
        super().__init__()
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding
        self.unfold = unfold
        self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias) #query plus user

        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x, incremental_state=None, query=None):
        '''Assuming the input, x, of the shape T x B x C and producing an output in the shape T x B x C
        args:
            x: Input of shape T x B x C, i.e. (timesteps, batch_size, input_size)
            incremental_state: A dict to keep the state
            unfold: unfold the input or not. If not, we use the matrix trick instead
            query: use the specified query to predict the conv filters
        '''
    
        query = x

        output = self._forward_unfolded(x, incremental_state, query)
        

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)

        return output

    def _forward_unfolded(self, x, incremental_state, query):
        '''The conventional implementation of convolutions.
        Unfolding the input by having a window shifting to the right.'''
        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H   
        
        assert R * H == C == self.input_size

        
        weight = self.weight_linear(query).view(T*B*H, -1) #generat user specific weights
        
        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)


        padding_l = self.padding_l
        if K > T and padding_l == K-1:
            weight = weight.narrow(1, K-T, T)
            K, padding_l = T, T-1
        # unfold the input: T x B x C --> T' x B x C x K
        x_unfold = unfold1d(x, K, padding_l, 0)
        x_unfold = x_unfold.view(T*B*H, R, K)

        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)
            
        weight = weight.narrow(1, 0, K)
      
        output = torch.bmm(x_unfold, weight.unsqueeze(2)) # T*B*H x R x 1
        output = output.view(T, B, C)
        return output

