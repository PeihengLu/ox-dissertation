from torch import nn
import torch
import pandas as pd

import sys
sys.path.append('../')
from typing import Dict

from utils.ml_utils import clones


class SequenceEmbedder(nn.Module):
    def __init__(self, embed_dim, annot_embed, assemb_opt='add'):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 2 # padding index for protospacer, PBS and RTT
        self.assemb_opt = assemb_opt
        # wt+mut+protospacer+PBS+RTT
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=0)
        # protospacer embedding
        self.Wproto = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # PBS embedding
        self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # RTT embedding
        self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
    
    def forward(self, X_nucl, X_proto, X_pbs, X_rt):
        if self.assemb_opt == 'add':
            return self.We(X_nucl) + self.Wproto(X_proto) + self.Wpbs(X_pbs) + self.Wrt(X_rt)
        elif self.assemb_opt == 'stack':
            return torch.cat([self.We(X_nucl), self.Wproto(X_proto), self.Wpbs(X_pbs), self.Wrt(X_rt)], axis=-1)


# feature processing
class MLPEmbedder(nn.Module):
    def __init__(self,
                 input_dim, # number of features
                 embed_dim, # number of features after embedding
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        self.We = nn.Linear(input_dim, embed_dim, bias=True)
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        self.encunit_pipeline = nn.Sequential(*encunit_layers)

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)
        return out

class MLPBlock(nn.Module):
            
    def __init__(self,
                 input_dim,
                 embed_dim,
                 mlp_embed_factor,
                 nonlin_func, 
                 pdropout):
        
        super().__init__()
        
        assert input_dim == embed_dim

        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, embed_dim*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_dim*mlp_embed_factor, embed_dim)
        )
        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, X):
        """
        Args:
            X: input tensor, (batch, sequence length, input_dim)
        """
        o = self.MLP(X)
        o = self.layernorm_1(o + X)
        o = self.dropout(o)
        return o

# meta learner for the RNN and MLP ensemble
class MLPDecoder(nn.Module):
    def __init__(self,
                 inp_dim,
                 embed_dim,
                 outp_dim,
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        self.We = nn.Linear(inp_dim, embed_dim, bias=True)
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        self.encunit_pipeline = nn.Sequential(*encunit_layers)

        self.W_mu = nn.Linear(embed_dim, outp_dim)
        # self.W_sigma = nn.Linear(embed_dim, outp_dim)
        
        # self.softplus = nn.Softplus()

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)

        mu = self.W_mu(out)
        return mu
        # logsigma  = self.W_sigma(out)
        # sigma = 0.1 + 0.9 * self.softplus(logsigma)

        # return mu, sigma
        

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    
def preprocess_transformer(X_train: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """transform the pridict data into a format that can be used by the model

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
    
    protospacer_location = X_train['protospacer-location'].values
    pbs_start = X_train['pbs-location-l-relative-protospacer'].values + protospacer_location
    rtt_start = X_train['rtt-location-l-relative-protospacer'].values + protospacer_location
    
    mut_type = X_train['mut-type'].values
    
    edit_length = X_train['edit-length'].values
    pbs_length = X_train['pbs-length'].values
    rtt_length = X_train['rtt-length'].values

    rtt_length_mut = []
    
    for i in range(len(wt_seq)):
        if mut_type[i] == 2:
            rtt_length_mut.append(rtt_length[i] - edit_length[i])
        elif mut_type[i] == 1:
            rtt_length_mut.append(rtt_length[i] + edit_length[i])
        else:
            rtt_length_mut.append(rtt_length[i])
        
    X_pbs = torch.zeros((len(wt_seq), len(wt_seq[0])))    
    X_rtt = torch.zeros((len(wt_seq), len(wt_seq[0])))    
    X_proto = torch.zeros((len(wt_seq), len(wt_seq[0])))
    X_rtt_mut = torch.zeros((len(wt_seq), len(wt_seq[0])))
    
    for i in range(len(wt_seq)):
        for j in range(int(pbs_start[i]), int(pbs_start[i]+pbs_length[i])):
            X_pbs[i, j] = 1
        for j in range(int(rtt_start[i]), int(rtt_start[i]+rtt_length[i])):
            X_rtt[i, j] = 1
        for j in range(int(rtt_start[i]), int(rtt_start[i]+rtt_length_mut[i])):
            X_rtt_mut[i, j] = 1
        for j in range(int(protospacer_location[i]), int(protospacer_location[i]+len(wt_seq[i]))):
            X_proto[i, j] = 1
        # annotate the padding regions
        for j in range(len(wt_seq[i])):
            if wt_seq[i][j] == 'N':
                X_pbs[i, j] = 2
                X_rtt[i, j] = 2
                X_proto[i, j] = 2
                X_rtt_mut[i, j] = 2
            
    nut_to_ix = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    X_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in wt_seq])
    X_mut_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in mut_seq])
    
    # transform to int64
    X_pbs = X_pbs.to(torch.int64)
    X_rtt = X_rtt.to(torch.int64)
    X_proto = X_proto.to(torch.int64)
    X_rtt_mut = X_rtt_mut.to(torch.int64)
    X_nucl = X_nucl.to(torch.int64)
    
    result = {
        'X_nucl': X_nucl,
        'X_proto': X_proto,
        'X_pbs': X_pbs,
        'X_rt': X_rtt,
        'X_mut_nucl': X_mut_nucl,
        'X_mut_pbs': X_pbs,
        'X_mut_rt': X_rtt_mut,
        'features': torch.tensor(features)
    }
    
    return result