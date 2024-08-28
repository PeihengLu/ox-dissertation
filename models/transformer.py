import skorch.helper
import skorch.scoring
from torch import nn
import torch
import pandas as pd
import numpy as np
import os
import skorch
import scipy
import math
from skorch.hf import AccelerateMixin
from accelerate import Accelerator
from skorch.helper import SliceDict
from sklearn.model_selection import ParameterGrid
import time
import torch.nn.functional as F
import copy
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from glob import glob

import sys
sys.path.append('../')
from typing import Dict, List, Tuple

from utils.ml_utils import clones, StackedTransformer
# from flash_attn import flash_attn_qkvpacked_func
# from local_attention import LocalAttention

class SequenceEmbedder(nn.Module):
    def __init__(self, embed_dim: int = 4, sequence_length=99, onehot: bool = True, annot: bool = False):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.onehot = onehot
        self.annot = annot
        # self.num_inidc = 2 # padding index for protospacer, PBS and RTT
        # wt+mut sequence embedding
        if not self.onehot:
            self.We = nn.Embedding(self.num_nucl+1, 4, padding_idx=self.num_nucl)
        # # protospacer embedding
        # self.Wproto = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # # PBS embedding
        # self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # # RTT embedding
        # self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        
        # Create a matrix of shape (max_len, embed_dim) for position encodings
        position_encoding = torch.zeros(sequence_length, embed_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term (10000^(2i/embed_dim))
        # This will be used to compute the sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra batch dimension to the position encoding
        position_encoding = position_encoding.unsqueeze(0)
        
        # Register the position encoding as a buffer, which is a tensor not considered a model parameter
        self.register_buffer('position_encoding', position_encoding)
    
    def forward(self, X_nucl: torch.tensor, X_pbs: torch.tensor=None, X_rtt: torch.tensor=None, padding_mask: torch.tensor=None) -> torch.tensor:
        """forward pass of the sequence embedder

        Args:
            X_nucl (torch.tensor): numerical representation of the nucleotide sequence
            padding_mask (torch.tensor, optional): tensor, float32, (batch, sequence length, embed_dim) representing the padding mask. Defaults to None.

        Returns:
            torch.tensor: tensor, float32, (batch, sequence length, embed_dim) embedded sequence
        """
        if self.onehot:
            # one hot encode the sequence
            x = F.one_hot(X_nucl, num_classes=self.num_nucl)
        else:
            x = self.We(X_nucl)

        if self.annot:
            # add a dimension to the PBS and RTT
            X_pbs = X_pbs.unsqueeze(-1)
            X_rtt = X_rtt.unsqueeze(-1)
            # concatenate the positional information
            x = torch.cat([x, X_pbs, X_rtt], dim=-1)
        
        # position embedding for non padding sequence using sinusoidal function
        x = x + self.position_encoding[:, :x.size(1)].requires_grad_(False)
        
        if padding_mask is not None:
            # Expand the mask to match the dimensions of x (batch_size, seq_len, embed_dim)
            # print distinct values in the padding mask
            x = x.masked_fill(padding_mask, 0)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, pdropout, flash: bool = False, local: bool = False):
        super(MultiHeadAttention, self).__init__()
        # embedding dimension of the sequence must be divisible by the number of heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(pdropout)
        self.attention = attention if not local else LocalAttention(window_size=3, causal=False, look_backward=1, look_forward=0, dropout=pdropout)
        self.attn = None
        self.flash = flash
        self.local = local
    
    def forward(self, query, key, value, mask = None):
        "Implement the scaled dot product attention"
        # query, key, value are all of shape (batch, sequence length, embed_dim)
        if mask is not None:
            # Same mask applied to all heads
            mask = mask.unsqueeze(1)
            # mask should be applied to the encoder sequence
            mask = mask.permute(0, 1, 3, 2)
                        
        batch_size = query.size(0)
        
        # Do all the linear projections in batch from embed_dim => num_heads x head_dim
        # split the embed_dim into num_heads
        # (batch, sequence length, embed_dim) => (batch, sequence length, num_heads, head_dim)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        
        if self.flash:
            # stack the query, key and value
            # (batch, sequence length, num_heads, head_dim) => (batch, num_heads, 3, sequence length, head_dim)
            qkv = torch.stack([query, key, value], dim=-1)
            # permute the dimensions to (batch, sequence length, 3, num_heads, head_dim)
            qkv = qkv.permute(0, 1, 4, 2, 3)
            
            if self.local:
                # feed the input to the flash attention
                x, softmax_lse, self.attn = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.dropout.p, return_attn_probs=True, window_size=(2, 2))
            else:
                x, softmax_lse, self.attn = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.dropout.p, return_attn_probs=True)
                
            del qkv
            # del softmax_lse
        # elif self.attention != attention: 
        #     # use local attention
        else:
            if self.local:
                # no attention probabilities is returned
                x = self.attention(query, key, value, mask=mask)
            else:
                x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # concatenate the output of the attention heads into the same shape as the input
        # (batch, sequence length, num_heads, head_dim) => (batch, sequence length, embed_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        del query, key, value
        
        return self.linears[-1](x)
    
# attention layer for pooling the transformer outputs
class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''
        # scale the input and query vector to prevent vanishing gradients
        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)

        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm
    
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_embed_dim, pdropout):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_embed_dim)
        self.linear2 = nn.Linear(mlp_embed_dim, embed_dim)
        self.dropout = nn.Dropout(pdropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)#.half()
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        # self attention
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, wt_mask: torch.Tensor, mut_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, enc_out, wt_mask, mut_mask)
        return self.norm(x)#.half()

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # self attention
        self.self_attn = self_attn
        # source attention
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout), 3)
        self.size = size
        
    def forward(self, x, enc_out, wt_mask, mut_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mut_mask))
        # wild type mask makes sure that the model does not attend to the padding values in the wild type sequence
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, enc_out, enc_out, wt_mask))
        return self.sublayer[2](x, self.feed_forward)

class Transformer(nn.Module):
    """encoder decoder transformer architecture

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, encoder, decoder, wt_embed, mut_embed, onehot=True, annot: bool = False):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.wt_embed = wt_embed
        self.mut_embed = mut_embed
        self.onehot = onehot
        self.annot = annot
        
    def forward(self, X_wt: torch.Tensor, X_mut: torch.Tensor, X_pbs, X_rtt, X_rtt_mut) -> torch.Tensor:
        """forward pass of the transformer

        Args:
            X_wt (torch.Tensor): one dimensional tensor representing the wild type sequence, (batch, sequence length)
            X_mut (torch.Tensor): one dimensional tensor representing the mutated sequence, (batch, sequence length)
            wt_mask (torch.Tensor): mask for the wild type sequence, (batch, sequence length)
            mut_mask (torch.Tensor): mask for the mutated sequence, (batch, sequence length) 

        Returns:
            _type_: _description_
        """
        padding_mask_wt = (X_wt == 4)
        padding_mask_mut = (X_mut == 4)
        
        if self.onehot:
            # convert the padding value to 0 so that one hot encoding can be applied
            X_wt = X_wt.masked_fill(padding_mask_wt, 0)
            X_mut = X_mut.masked_fill(padding_mask_mut, 0)
            
        # the mask should mask all embed dimensions
        padding_mask_wt = padding_mask_wt.unsqueeze(-1)
        padding_mask_mut = padding_mask_mut.unsqueeze(-1)
                
        # convert the sequence to embeddings
        # (batch, sequence length, embed_dim)
        if self.annot:
            wt_embed = self.wt_embed(X_nucl=X_wt, padding_mask = padding_mask_wt, X_pbs=X_pbs, X_rtt=X_rtt)
            mut_embed = self.mut_embed(X_nucl=X_mut, padding_mask = padding_mask_mut, X_pbs=X_pbs, X_rtt=X_rtt_mut) 
        else:
            wt_embed = self.wt_embed(X_wt, padding_mask = padding_mask_wt)
            mut_embed = self.mut_embed(X_mut, padding_mask = padding_mask_mut) 
        
        # transform to half precision for faster computation
        wt_embed = wt_embed#.half()
        mut_embed = mut_embed#.half()
        
        padding_mask_wt = padding_mask_wt#.half()
        padding_mask_mut = padding_mask_mut#.half()
        
        # print('wt_embed:', wt_embed.dtype)
        
        # wt_seq and mut_seq are already masked
        enc_out = self.encoder(wt_embed, padding_mask_wt)
        dec_out = self.decoder(mut_embed, enc_out, padding_mask_wt, padding_mask_mut)
        
        return dec_out

def make_model(N=6, embed_dim=4, mlp_embed_dim=64, num_heads=4, pdropout=0.1, onehot=True, annot=False, flash=False, local=False):
    c = copy.deepcopy
    attn = MultiHeadAttention(num_heads, embed_dim, pdropout, flash=flash)
    attn_local = MultiHeadAttention(num_heads, embed_dim, pdropout, local=True, flash=flash) if local else attn
    position_ff = PositionwiseFeedForward(embed_dim, mlp_embed_dim, pdropout)
    model = Transformer(
        encoder=Encoder(EncoderLayer(embed_dim, c(attn_local), c(position_ff), pdropout), N),
        decoder=Decoder(DecoderLayer(embed_dim, c(attn_local), c(attn), c(position_ff), pdropout), N),
        wt_embed=SequenceEmbedder(embed_dim=embed_dim, onehot=onehot, annot=annot),
        mut_embed=SequenceEmbedder(embed_dim=embed_dim, onehot=onehot, annot=annot),
        annot=annot,
        onehot=onehot
    )
    
    return model

class PrimeDesignTransformer(nn.Module):
    def __init__(self, sequence_length=99, pdropout=0.1, mlp_embed_dim=100, num_encoder_units=1, num_features=24, flash=False, onehot=True, annot=False, local=False):
        super(PrimeDesignTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.pdropout = pdropout
        self.mlp_embed_dim = mlp_embed_dim
        self.num_encoder_units = num_encoder_units
        self.num_features = num_features
        self.flash = flash
        self.onehot = onehot
        self.attention_values = [0 for _ in range(sequence_length)]
        
        self.embed_dim = 6 if annot else 4
        self.num_heads = 3 if annot else 2
                
        self.transformer = make_model(N=num_encoder_units, embed_dim=self.embed_dim, mlp_embed_dim=mlp_embed_dim, num_heads=self.num_heads, pdropout=pdropout, onehot=onehot, annot=annot, flash=flash, local=local)
        self.transformer_pool = FeatureEmbAttention(input_dim=self.embed_dim)
        
        # self.gru = nn.GRU(input_size=sequence_length, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(pdropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(pdropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim + 128),
            nn.Dropout(pdropout),
            nn.Linear(self.embed_dim + 128, 1, bias=True),
        )
        
        # self.generator = nn.Sequential(
        #     nn.Linear(embed_dim, 1, bias=False),            
        # )

        # initialize the parameters with xavier uniform
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        
    def forward(self, X_nucl: torch.Tensor, X_mut_nucl: torch.Tensor, X_pbs, X_rtt, X_rtt_mut, features: torch.Tensor) -> torch.Tensor:
        """forward pass of the transformer model

        Args:
            X_nucl (torch.Tensor): tensor, float32, (batch, sequence length) representing the wild type sequence
            X_mut_nucl (torch.Tensor): tensor, float32, (batch, sequence length) representing the mutated sequence
            features (torch.Tensor): tensor, float32, (batch, num_features) representing the features

        Returns:
            torch.Tensor: tensor, float32, (batch, 1) representing the predicted value
        """
        # print('X_nucl shape:', X_nucl.shape)
        # print('X_mut_nucl shape:', X_mut_nucl.shape)
        # print('features shape:', features.shape)
        
        # convert the sequence to embeddings
        # (batch, sequence length, embed_dim)
        transformer_out = self.transformer(X_nucl, X_mut_nucl, X_pbs, X_rtt, X_rtt_mut)
        
        # reduce the sequence to its dimension
        # (batch, sequence length, embed_dim) => (batch, embed_dim)
        # reshape so that the linear layer can be applied to each of the dimensions
        transformer_out, self.attention_values = self.transformer_pool(transformer_out)
        # # print('transformer_out shape:', transformer_out.shape)
        
        # convert the features to embeddings
        # (batch, sequence length, embed_dim)
        features_embed = self.feature_embedding(features)
                
        # concatenate the output of the transformer and the features
        # (batch, sequence length, embed_dim)
        output = torch.cat([transformer_out, features_embed], dim=1)
        
        # print('output shape:', output.shape)
        
        # pass the output to the MLP decoder
        output = self.head(output)
        
        del transformer_out, features_embed, X_nucl, X_mut_nucl, X_pbs, X_rtt, X_rtt_mut, features
        
        # print('output shape:', output.shape)
        # print('output:', output)
        
        # return self.generator(transformer_out)
        
        return output
    
    
class AcceleratedNet(AccelerateMixin, skorch.NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def preprocess_transformer(X_train: pd.DataFrame, slice: bool=False) -> Dict[str, torch.Tensor]:
    """transform the transformer data into a format that can be used by the model

    Args:
        X_train (pd.DataFrame): the sequence and feature level data

    Returns:
        Dict[str, torch.Tensor]: dictionary of input names and their corresponding tensors, so that skorch can use them with the forward function
    """
    # sequence data
    wt_seq = X_train['wt-sequence'].values
    mut_seq = X_train['mut-sequence'].values
    # the rest are the features
    features = X_train.iloc[:, 2:26].values

    nut_to_ix = {'N': 4, 'A': 0, 'T': 1, 'G': 2, 'C': 3}
    X_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in wt_seq])
    X_mut_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in mut_seq])
    # create a linear embedding for pbs, protospacer, and rtt values
    X_pbs = torch.zeros(X_nucl.size(0), X_nucl.size(1))
    # X_protospacer = torch.zeros(X_nucl.size(0), X_nucl.size(1))
    X_rtt = torch.zeros(X_nucl.size(0), X_nucl.size(1))
    X_rtt_mut = torch.zeros(X_nucl.size(0), X_nucl.size(1))
    
    for i, (pbs_l, protospacer_l, rtt_l, rtt_mut_l, pbs_r, protospacer_r, rtt_r, rtt_mut_r) in enumerate(zip(X_train['pbs-location-l'].values, X_train['protospacer-location-l'].values, X_train['rtt-location-l'].values, X_train['rtt-location-l'].values, X_train['pbs-location-r'].values, X_train['protospacer-location-r'].values, X_train['rtt-location-r'].values, X_train['rtt-location-r'].values)):
        pbs_l = max(0, pbs_l)
        pbs_r = max(0, pbs_r)
        protospacer_l = max(0, protospacer_l)
        protospacer_r = max(0, protospacer_r)
        rtt_l = max(0, rtt_l)
        rtt_r = max(0, rtt_r)
        X_pbs[i, pbs_l:pbs_r] = 1
        # X_protospacer[i, protospacer_l:protospacer_r] = 1
        X_rtt[i, rtt_l:rtt_r] = 1
        X_rtt_mut[i, rtt_mut_l:rtt_mut_r] = 1
    
    result = {
        'X_nucl': X_nucl,
        'X_mut_nucl': X_mut_nucl,
        'X_pbs': X_pbs,
        # 'X_protospacer': X_protospacer,
        'X_rtt': X_rtt,
        'X_rtt_mut': X_rtt_mut,
        'features': torch.tensor(features).float()#.half()
    }

    
    return result

def train_transformer(train_fname: str, lr: float, batch_size: int, epochs: int, patience: int, num_runs: int, num_features: int, dropout: float = 0.1, percentage: str = 1, annot: bool = False, num_encoder_units: int = 1, onehot: bool=True) -> skorch.NeuralNetRegressor:
    """train the transformer model

    Args:
        train_fname (str): the name of the csv file containing the training data
        lr (float): learning rate
        batch_size (int): batch size
        epochs (int): number of epochs
        patience (int): number of epochs to wait before early stopping
        num_runs (int): number of repeated runs on one fold
        num_features (int): number of features to use for the MLP
        adjustment (str, optional): adjustment to the target value. Defaults to 'None'.
        dropout (float, optional): percentage of input units to drop. Defaults to 0.1.
        percentage (str, optional): percentage of the training data to use. Defaults to 1, meaning all the data will be used.

    Returns:
        skorch.NeuralNetRegressor: _description_
    """
    # load a dp dataset
    dp_dataset = pd.read_csv(os.path.join('models', 'data', 'transformer', train_fname))
    
    # remove rows with nan values
    dp_dataset = dp_dataset.dropna()
    
    sequence_length = len(dp_dataset['wt-sequence'].values[0])
    
    # if percentage is less than 1, then use a subset of the data
    if percentage < 1:
        dp_dataset = dp_dataset.sample(frac=percentage, random_state=42)
    
    # TODO read the top 2000 rows only during development
    # dp_dataset = dp_dataset.head(2000)
    
    # data origin
    data_origin = os.path.basename(train_fname).split('-')[1]
    
    fold = 5
        
    # device
    device = torch.device('cuda')
    
    for i in range(1, fold):
        print(f'Fold {i+1} of {fold}')
        
        train = dp_dataset[dp_dataset['fold']!=i]
        X_train = train
        print(X_train.columns)
        y_train = train.iloc[:, -2]

        X_train = preprocess_transformer(X_train)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        # check if X_train contains nan values
        if torch.isnan(X_train['X_nucl']).any():
            print('X_nucl contains nan values')
        if torch.isnan(X_train['X_mut_nucl']).any():
            print('X_mut_nucl contains nan values')
        if torch.isnan(X_train['features']).any():
            print('features contains nan values')
        
        print("Training Transformer model...")
        
        best_val_loss = np.inf
    
        for j in range(num_runs):
            print(f'Run {j+1} of {num_runs}')
            # model
            m = PrimeDesignTransformer(sequence_length=sequence_length, pdropout=dropout, num_features=num_features, onehot=onehot, annot=annot, flash=False, local=False, num_encoder_units=num_encoder_units)
            
            # accelerator = Accelerator(mixed_precision='bf16')
            
            # skorch model
            model = skorch.NeuralNetRegressor(
                m,
                # accelerator=accelerator,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                # optimizer__eps=1e-4,
                # optimizer=torch.optim.SGD,
                optimizer__lr=lr,
                device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                batch_size=batch_size,
                max_epochs=epochs,
                train_split= skorch.dataset.ValidSplit(cv=5),
                # early stopping
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=patience),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', 
                                    f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), 
                                    f_optimizer=None, 
                                    f_history=None,
                                    f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=15, T_mult=1, eta_min=1e-6),
                    # skorch.callbacks.ProgressBar(),
                    # skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau, monitor='valid_loss', factor=0.5, patience=3, min_lr=1e-6),
                    # PrintParameterGradients()
                ]
            )
            
            # model.initialize()
            # torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            
            model.fit(X_train, y_train)
            
            if np.min(model.history[:, 'valid_loss']) < best_val_loss:
                print(f'Best validation loss: {np.min(model.history[:, "valid_loss"])}')
                best_val_loss = np.min(model.history[:, 'valid_loss'])
                # rename the model file to the best model
                os.rename(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
            else: # delete the last model
                print(f'Validation loss: {np.min(model.history[:, "valid_loss"])} is not better than {best_val_loss}')
                os.remove(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"))
            
        
    return model

def predict(test_fname: str, num_features: int, adjustment: str = None, device: str = 'cuda', dropout: float=0, percentage: float = 1.0, annot: bool = False) -> skorch.NeuralNetRegressor:
    # model name
    fname = os.path.basename(test_fname)
    model_name =  fname.split('.')[0]
    model_name = '-'.join(model_name.split('-')[1:])
    models = [os.path.join('models', 'trained-models', 'transformer', f'{model_name}-fold-{i}.pt') for i in range(1, 6)]
    # Load the data
    test_data_all = pd.read_csv(os.path.join('models', 'data', 'transformer', test_fname))    
    # if percentage is less than 1, then use a subset of the data
    if percentage < 1:
        test_data_all = test_data_all.sample(frac=percentage, random_state=42)
    # remove rows with nan values
    test_data_all = test_data_all.dropna()
    # transform to float
    test_data_all.iloc[:, 2:26] = test_data_all.iloc[:, 2:26].astype(float)
    
    sequence_length = len(test_data_all['wt-sequence'].values[0])

    m = PrimeDesignTransformer(sequence_length=sequence_length, pdropout=dropout, num_encoder_units=3, num_features=num_features, onehot=True, annot=annot, flash=False, local=False)
    
    accelerator = Accelerator(mixed_precision='bf16')
            
    # skorch model
    tr_model = AcceleratedNet(
        m,
        accelerator=accelerator,
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
    )

    prediction = {}
    performance = []

    # Load the models
    for i, model in enumerate(models):
        if not os.path.isfile(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt")):
            continue
        
        test_data = test_data_all[test_data_all['fold']==i]
        X_test = test_data
        y_test = test_data.iloc[:, -2]
        X_test = preprocess_transformer(X_test)
        y_test = y_test.values
        y_test = y_test.reshape(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        tr_model.initialize()
        if adjustment:
            tr_model.load_params(f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
        else:
            tr_model.load_params(f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
        
        y_pred = tr_model.predict(X_test)
        if adjustment == 'log':
            y_pred = np.expm1(y_pred)

        pearson = np.corrcoef(y_test.T, y_pred.T)[0, 1]
        spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

        print(f'Fold {i + 1} Pearson: {pearson}, Spearman: {spearman}')

        prediction[i] = y_pred
        performance.append((pearson, spearman))
    
    return prediction


from sklearn.model_selection import ParameterGrid
import time
import scipy.stats

from sklearn.model_selection import ParameterGrid
import time
import scipy.stats

def tune_transformer(tune_fname: str, lr: float, batch_size: int, epochs: int, patience: int, num_runs: int, num_features: int, dropout: float = 0.1, percentage: str = 1) -> skorch.NeuralNetRegressor:
    """perform hyperparameter tuning for the transformer model

    Args:
        tune_fname (str): the name of the csv file containing the test data
        num_features (int): number of features to use for the MLP
        adjustment (str, optional): adjustment to the target value. Defaults to 'None'.
        device (str, optional): device used for tuning. Defaults to 'cuda'.
        dropout (float, optional): percentage of input units to drop. Defaults to 0.1.
        percentage (float, optional): percentage of the training data to use. Defaults to 1.0.
        num_runs (int, optional): number of repeated runs on one fold. Defaults to 5.
    """
    # using gridsearchcv for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV

    params_arch = {
        'module__num_encoder_units': [1, 3, 5],
        'module__pdropout': [0.05, 0.1, 0.2, 0.3, 0.5], # 0.01, 0.05,
        'module__mlp_embed_dim': [50, 100, 150],
        # 'module__pdropout': [0.1, 0.3, 0.5],
        # 'module__flash': [True, False],
        # 'module__local': [True, False],
        # 'module__annot': [True, False],
    }
    # for a grid using the full parameter space
    # list of lists of parameters
    params_arch = list(ParameterGrid(params_arch))

    # load a dp dataset
    dp_dataset = pd.read_csv(os.path.join('models', 'data', 'transformer', tune_fname))

    # remove rows with nan values
    dp_dataset = dp_dataset.dropna()

    # if percentage is less than 1, then use a subset of the data
    if percentage < 1:
        dp_dataset = dp_dataset.sample(frac=percentage, random_state=42)

    train_data = dp_dataset[dp_dataset['fold']!=0]
    test_data = dp_dataset[dp_dataset['fold']==0]
    X_train = train_data
    y_train = train_data.iloc[:, -2]
    X_test = test_data
    y_test = test_data.iloc[:, -2]

    X_train = preprocess_transformer(X_train)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test = preprocess_transformer(X_test)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    fname = 'transformer-train-fine-tune'

    # use fold 0 for tuning
    for ind, par in enumerate(params_arch):
        performances = os.path.join('models', 'data', 'performance', f'{fname}.csv')
        # check if the parameter has already been tuned
        if os.path.isfile(performances):
            performances = pd.read_csv(performances)
            # convert the dataframe to a dictionary
            row = performances[(performances['module__num_encoder_units'] == par['module__num_encoder_units']) & (performances['module__pdropout'] == par['module__pdropout']) & (performances['module__mlp_embed_dim'] == par['module__mlp_embed_dim'])]
            if len(row['pearson'].isna().values) > 0 and not row['pearson'].isna().values[0]:
                print(f'Parameter: {par} has already been tuned')
                print('-'*50, '\n')
                par['pearson'] = row['pearson'].values[0]
                par['spearman'] = row['spearman'].values[0]
                continue
    for ind, par in enumerate(params_arch):
        performances = os.path.join('models', 'data', 'performance', f'{fname}.csv')
        # check if the parameter has already been tuned
        if os.path.isfile(performances):
            if 'pearson' in par:
                print(f'Parameter: {par} has already been tuned')
                print('-'*50, '\n')
                continue

        performances_pearson = []
        performances_spearman = []
        print(f'Parameter: {par}')
        for run in range(num_runs):
            t = time.time()
            print(f'Run {run+1} of {num_runs}')
            accelerator = Accelerator(mixed_precision='bf16')

            save_file_name = os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(tune_fname).split('.')[0].split('-')[1:])}-run-{run+1}")
            # concatenate the parameter to the save file name
            for p in par:
                save_file_name += f"-{p}-{par[p]}"
            save_file_name += '.pt'
            
            print(f'Save file name: {save_file_name}')

            # skorch model
            model = skorch.NeuralNetRegressor(
                PrimeDesignTransformer,
                module__sequence_length=99,
                module__pdropout=dropout,
                module__num_encoder_units=3,
                module__num_features=num_features,
                module__flash=False,
                module__local=False,
                module__annot=True,
                module__mlp_embed_dim=100,
                # accelerator=accelerator,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                # optimizer__eps=1e-4,
                # optimizer=torch.optim.SGD,
                optimizer__lr=0.0025,
                max_epochs=500,
                device='cuda',
                batch_size=2048,
                train_split= skorch.dataset.ValidSplit(cv=5),
                # early stopping
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=patience),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=15, T_mult=1, eta_min=1e-5),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=save_file_name, f_optimizer=None, f_history=None, f_criterion=None),
                    # skorch.callbacks.ProgressBar(),
                    # PrintParameterGradients()
                ]
            )
            model.set_params(**par)

            model.fit(X_train, y_train)

            # evaluate the model
            model = skorch.NeuralNetRegressor(
                PrimeDesignTransformer,
                module__sequence_length=99,
                module__pdropout=0,
                module__num_encoder_units=3,
                module__num_features=num_features,
                module__flash=False,
                module__local=False,
                module__annot=True,
                module__mlp_embed_dim=100,
                # accelerator=accelerator,
                device='cuda',
                batch_size=512,
                criterion=nn.MSELoss,
            )

            model.set_params(**par)
            # drop out should still be 0
            model.set_params(module__pdropout=0)
            model.initialize()

            model.load_params(f_params=save_file_name)

            y_pred = model.predict(X_test)
            pearson = np.corrcoef(y_test.T, y_pred.T)[0, 1]
            spearman = scipy.stats.spearmanr(y_test, y_pred)[0]
            
            print(f'Configuration: {par}, Run {run}, Pearson: {pearson}, Spearman: {spearman}')

            performances_pearson.append(pearson)
            performances_spearman.append(spearman)

            torch._C._cuda_clearCublasWorkspaces()
            torch._dynamo.reset()
            del model, accelerator
            torch.cuda.empty_cache()
            gc.collect()
            # delete the temporary model
            print(f'Run time: {time.time() - t}')
        par['pearson'] = performances_pearson
        par['spearman'] = performances_spearman

        print(f'Parameter: {par}')
        print('-'*50, '\n')

        # perform paired t test on the different configurations
        # and calculate the mean performance of each configuration
        performances = pd.DataFrame(params_arch)
        # save the performance
        performances.to_csv(os.path.join('models', 'data', 'performance', f'{fname}.csv'), index=False)
        
def fine_tune_transformer(fine_tune_fname: str = None):
    # load the fine tune datasets
    if not fine_tune_fname:
        fine_tune_data = glob(os.path.join('models', 'data', 'deepprime', '*small*.csv'))
    else:
        fine_tune_data = [fine_tune_fname]

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    for data in fine_tune_data:
        data_source = os.path.basename(data).split('-')[1:]
        data_source = '-'.join(data_source)
        data_source = data_source.split('.')[0]
        # load the fine tune data
        fine_tune_data = pd.read_csv(data)
        sequence_length = len(fine_tune_data['wt-sequence'].values[0])
        for i in range(5):
            fine_tune = fine_tune_data[fine_tune_data['fold'] != i]
            fold = i + 1
            # load the dp hek293t pe 2 model
            model = PrimeDesignTransformer(sequence_length=sequence_length, pdropout=0.2, num_encoder_units=1, num_features=24, flash=False, local=False, annot=True, mlp_embed_dim=100)
            model.load_state_dict(torch.load('models/trained-models/transformer/dp-hek293t-pe2-fold-1.pt', map_location=device))
            
            # freeze the layers other than head and feature mlps
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
            for param in model.feature_embedding.parameters():
                param.requires_grad = True
                
            # skorch wrapper
            # skorch model
            tr_model = skorch.NeuralNetRegressor(
                model,
                # accelerator=accelerator,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                # optimizer__eps=1e-4,
                # optimizer=torch.optim.SGD,
                optimizer__lr=0.001,
                max_epochs=500,
                device=device,
                batch_size=2048,
                train_split= skorch.dataset.ValidSplit(cv=5),
                # early stopping
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=20),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=15, T_mult=1, eta_min=1e-5),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'models/trained-models/transformer/dp-{data_source}-fold-{fold}.pt', f_optimizer=None, f_history=None, f_criterion=None),
                    # skorch.callbacks.ProgressBar(),
                    # PrintParameterGradients()
                ]
            )
            
            
            y_fine_tune = fine_tune.iloc[:, -2]
            X_fine_tune = preprocess_transformer(fine_tune)
            y_fine_tune = y_fine_tune.values
            y_fine_tune = y_fine_tune.reshape(-1, 1)
            y_fine_tune = torch.tensor(y_fine_tune, dtype=torch.float32)
            
            # train the model
            tr_model.fit(X_fine_tune, y_fine_tune)


def visualize_attention(model_name: str = 'dp-hek293t-pe2', num_features: int = 24, device: str = 'cuda', dropout: float=0, percentage: float = 1.0, annot: bool = False, num_encoder_units: int=1, onehot: bool =True) -> None:
    """visualize the attention weights of the transformer model

    Args:
        model (skorch.NeuralNetRegressor): the trained transformer model
        X_test (pd.DataFrame): the test data
        num_features (int): number of features to use for the MLP
        device (str, optional): device used for tuning. Defaults to 'cuda'.
        dropout (float, optional): percentage of input units to drop. Defaults to 0.1.
        percentage (float, optional): percentage of the training data to use. Defaults to 1.0.
    """
    # Load the data
    test_data_all = pd.read_csv(os.path.join('models', 'data', 'transformer', f'transformer-{model_name}.csv'))
    # if percentage is less than 1, then use a subset of the data
    if percentage < 1:
        test_data_all = test_data_all.sample(frac=percentage, random_state=42)
    test_data_all = test_data_all[test_data_all['fold']==0]
    # group the test data with edit length of 1 by edit type
    test_data_all = test_data_all[test_data_all['edit-length']==1]
    test_data_replace = test_data_all[test_data_all['mut-type']==0]
    test_data_insertion = test_data_all[test_data_all['mut-type']==1]
    test_data_deletion = test_data_all[test_data_all['mut-type']==2]
    # remove rows with nan values
    test_data_all = test_data_all.dropna()
    # transform to float
    test_data_all.iloc[:, 2:26] = test_data_all.iloc[:, 2:26].astype(float)
    
    embed_dim = 4 if not annot else 6
    num_heads = 3 if annot else 2

    sequence_length = len(test_data_all['wt-sequence'].values[0])

    m = PrimeDesignTransformer(embed_dim=embed_dim, sequence_length=sequence_length, num_heads=num_heads,pdropout=dropout, num_encoder_units=num_encoder_units, num_features=num_features, onehot=onehot, annot=annot, flash=False)
    
    accelerator = Accelerator(mixed_precision='bf16')
            
    # skorch model
    tr_model = skorch.NeuralNetRegressor(
        m,
        # accelerator=accelerator,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
    )

    tr_model.initialize()
    tr_model.load_params(f_params=os.path.join('models', 'trained-models', 'transformer', f'{model_name}-fold-1.pt'))

    prediction = {}
    performance = []

    attention_replace = [[[np.zeros(sequence_length) for _ in range(sequence_length)] for i in range(num_heads)] for j in range(num_encoder_units)]
    attention_insertion = [[[np.zeros(sequence_length) for _ in range(sequence_length)] for i in range(num_heads)] for j in range(num_encoder_units)]
    attention_deletion = [[[np.zeros(sequence_length) for _ in range(sequence_length)] for i in range(num_heads)] for j in range(num_encoder_units)]

    attentions = [attention_replace, attention_insertion, attention_deletion]

    count_replace = [0 for _ in range(sequence_length)]
    count_insertion = [0 for _ in range(sequence_length)]
    count_deletion = [0 for _ in range(sequence_length)]

    counts = [count_replace, count_insertion, count_deletion]

    # Load the models
    for i, data in enumerate([test_data_replace, test_data_insertion, test_data_deletion]):
        # run one example after each other and acquire the attention value
        # corresponding to the edit position
        X_test = preprocess_transformer(data)
        for ind in tqdm.tqdm(range(len(data))):
            item = {}
            for key in X_test:
                item[key] = X_test[key][ind].unsqueeze(0)
            data_item = data.iloc[ind]
            # print(f'Edit type: {data_item["mut-type"]}, Edit position: {data_item["lha-location-r"]}')
            y_pred = tr_model.predict(item)
            
            for layer in range(num_encoder_units):
                # get the attention weights
                attns = tr_model.module_.transformer.decoder.layers[layer].cross_attn.attn

                # plot the attention weights for each head
                # for j in range(num_heads):
                #     plt.figure(figsize=(10, 10))
                #     # highlight the diagonal
                #     sns.heatmap(attns[0, j, :, :].detach().cpu().numpy(), cmap='viridis')
                #     plt.title(f'Attention weights for head {j} and edit type {data_item["mut-type"]}')
                #     plt.show()
                #     break
                # print(attns.shape)
                
                # get the attention weights for each layer and head
                for j in range(num_heads):
                    normalized_attn = attns[0, j, data_item['lha-location-r'], :].detach().cpu().numpy()
                    # normalized_attn /= np.sum(normalized_attn)
                    attentions[i][layer][j][data_item['lha-location-r']] += normalized_attn
                    counts[i][data_item['lha-location-r']] += 1

    # normalize the attention weights by the number of times the attention weights were added
    for i in range(3):
        for j in range(num_heads):
            for k in range(sequence_length):
                for layer in range(num_encoder_units):
                    attentions[i][layer][j][k] /= counts[i][k]

    # plot the attention weights for each head and edit type
    for i, data in enumerate(['replace', 'insertion', 'deletion']):
        for j in range(num_heads):
            for layer in range(num_encoder_units):
                plt.figure(figsize=(10, 10))
                # highlight the diagonal
                sns.heatmap(attentions[i][layer][j], cmap='viridis')
                plt.title(f'Attention weights for head {j} and edit type {data} at layer {layer}')
                plt.show()

    return attentions