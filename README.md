Implementation of dynamic convolution for next item recommendation

Instructions for some important parameters:

'embed_dim' is set differently than dimension for feedforforward network 'ffn_embed_dim'

there are two dropouts: 'dropout' for dropout in FFN and 'weight_dropout' for FFN that changes kernel dynamically
for now both of them are set to 0.0 

'layers' denotes the number of layers for dynamic convolution and 'heads' denotes how many ways we are dividing the embedding dimention for applying convolution

'decoder_kernel_size_list' a list of number of kernels for each layer. For example, if layer == 2 then decoder_kernel_size_list =[5,5]. If only one number is given but number of layer is greater than one then the list is multiplied by the number of the layer (e.g. [5] will be expandded as [5,5])

'num_neg_samples' denotes number of negative examples for training (default = 100)
