Sourcecode for the paper: "DynamicRec: A Dynamic Convolutional Network for Next Item Recommendation". Please cite the paper if you find the code useful.

## Citation:
```
@inproceedings{tanjim2020dynamicrec,
  title={DynamicRec: A Dynamic Convolutional Network for Next Item Recommendation.},
  author={Tanjim, Md Mehrab and Ayyubi, Hammad A and Cottrell, Garrison W},
  booktitle={CIKM},
  pages={2237--2240},
  year={2020}
}
```

## Requirements:
Pytorch: Version >= 1.4.0

Python: Version >= 3.6

## Instructions: 

To run with default parameters, simply do: "python main.py"

If you want to experiment with changing other parameters, following are the important parameters to change:

```
'--dataset': Name of the dataset, the following are provided - 'nowplaying', 'diginetica', 'last_fm' (default), 'yoochoose'
'--top_k': Top k prediction for calculating HitRate@k, and NDCG@k (default=10)
'--batch_size': Default=128
'--maxlen': Maximum length of the sequence (default=30)
'--embed_dim': Embedding dimension (default=200)
'--ffn_embed_dim': Embedding dimension for Feedforward Network (default=200)
'--dropout': Dropout in FFN (default=0.2) 
'--weight_dropout': Dropout in FFN for dynamic convolution (default=0.2)
'--layers': Number of convolution layers (default=2) 
'--decoder_kernel_size_list': Kernel size in each of the convolution layer (default = [5, 5]). Note each entry in this list correspoend to kernel size of each convolution layer
'--num_neg_samples': Number of negative samples for calculating loss (default = 400)
```
