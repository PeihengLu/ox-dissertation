# PRIDICT Model by Gerald et al 2023
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from typing import Dict
import skorch
import scipy

import os
from glob import glob
from sklearn.preprocessing import StandardScaler

class RNN_Net(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 z_dim, 
                 device,
                 num_hiddenlayers=1, 
                 bidirection= False, 
                 rnn_pdropout=0., 
                 rnn_class=nn.LSTM, 
                 nonlinear_func=nn.ReLU(),
                 fdtype = torch.float32):
        
        super().__init__()
        self.fdtype = fdtype
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_hiddenlayers = num_hiddenlayers
        self.rnn_pdropout = rnn_pdropout
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.rnninput_dim = self.input_dim

        if num_hiddenlayers == 1:
            rnn_pdropout = 0
        self.rnn = rnn_class(self.rnninput_dim, 
                             hidden_dim, 
                             num_layers=num_hiddenlayers, 
                             dropout=rnn_pdropout, 
                             bidirectional=bidirection,
                             batch_first=True)
        if(bidirection):
            self.num_directions = 2
        else:
            self.num_directions = 1
   
        self.Wz = nn.Linear(self.num_directions*hidden_dim, self.z_dim)
        self.nonlinear_func = nonlinear_func    

        
    def init_hidden(self, batch_size, requires_grad=True):
        """
        initialize hidden vectors at t=0
        
        Args:
            batch_size: int, the size of the current evaluated batch
        """
        device = self.device
        # a hidden vector has the shape (num_layers*num_directions, batch, hidden_dim)
        h0=torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype)
        h0.requires_grad=requires_grad
        h0 = h0.to(device)
        if(isinstance(self.rnn, nn.LSTM)):
            c0=torch.zeros(self.num_hiddenlayers*self.num_directions, batch_size, self.hidden_dim).type(self.fdtype)
            c0.requires_grad=requires_grad
            c0 = c0.to(device)
            hiddenvec = (h0,c0)
        else:
            hiddenvec = h0
        return(hiddenvec)
    
    def forward_tbptt(self, trunc_batch_seqs, hidden):
        # run truncated backprop
        trunc_rnn_out, hidden = self.rnn(trunc_batch_seqs, hidden)

        z_logit = self.nonlinear_func(self.Wz(trunc_rnn_out))
            
        return (hidden, z_logit)
    
    def detach_hiddenstate_(self, hidden):
        # check if hidden is not tuple # case of GRU or vanilla RNN
        if not isinstance(hidden, tuple):
            hidden.detach_()
        else: # case of LSTM
            for s in hidden:
                s.detach_()
    
    def forward_complete(self, batch_seqs, seqs_len, requires_grad=True):
        """ perform forward computation
        
            Args:
                batch_seqs: tensor, shape (batch, seqlen, input_dim)
                seqs_len: tensor, (batch,), comprising length of the sequences in the batch
        """

        # init hidden
        hidden = self.init_hidden(batch_seqs.size(0), requires_grad=requires_grad)
        # pack the batch
        packed_embeds = pack_padded_sequence(batch_seqs, seqs_len.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_rnn_out, hidden = self.rnn(packed_embeds, hidden)

        # we need to unpack sequences
        unpacked_output, out_seqlen = pad_packed_sequence(packed_rnn_out, batch_first=True)
            
        z_logit = self.nonlinear_func(self.Wz(unpacked_output))
  
        return (hidden, z_logit)
    
    def forward(self, batch_seqs, seqs_len, requires_grad=True):
        return self.forward_complete(batch_seqs, seqs_len, requires_grad=requires_grad)


class MaskGenerator():
    def __init__(self):
        pass
    @classmethod
    def create_content_mask(clss, x_mask_shape, x_len):
        """
        Args:
            x_mask_shape: tuple, (bsize, max_seqlen)
            x_len: tensor, (bsize,), length of each sequence
        """
        x_mask = torch.ones(x_mask_shape)
        for bindx, tlen in enumerate(x_len):
            x_mask[bindx, tlen:] = 0
        return x_mask

# wt sequence embedding
class AnnotEmbeder_WTSeq(nn.Module):
    def __init__(self, embed_dim, annot_embed_dim, assemb_opt='add'):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 2 # padding index for protospacer, PBS and RTT
        self.assemb_opt = assemb_opt
        # wt+mut+protospacer+PBS+RTT
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=self.num_nucl)
        # protospacer embedding
        self.Wproto = nn.Embedding(self.num_inidc+1, annot_embed_dim, padding_idx=self.num_inidc)
        # PBS embedding
        self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed_dim, padding_idx=self.num_inidc)
        # RTT embedding
        self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed_dim, padding_idx=self.num_inidc)
    
    def forward(self, X_nucl, X_proto, X_pbs, X_rt):
        if self.assemb_opt == 'add':
            return self.We(X_nucl) + self.Wproto(X_proto) + self.Wpbs(X_pbs) + self.Wrt(X_rt)
        elif self.assemb_opt == 'stack':
            return torch.cat([self.We(X_nucl), self.Wproto(X_proto), self.Wpbs(X_pbs), self.Wrt(X_rt)], axis=-1)


# mutated sequence embedding
class AnnotEmbeder_MutSeq(nn.Module):
    def __init__(self, embed_dim, annot_embed_dim, assemb_opt='add'):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 2 # padding index
        self.assemb_opt = assemb_opt
        # one hot encoding
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=self.num_nucl)
        # PBS embedding
        self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed_dim, padding_idx=self.num_inidc)
        # RTT embedding
        self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed_dim, padding_idx=self.num_inidc)
    
    def forward(self, X_nucl, X_pbs, X_rt):
        if self.assemb_opt == 'add':
            return self.We(X_nucl) + self.Wpbs(X_pbs) + self.Wrt(X_rt)
        elif self.assemb_opt == 'stack':
            return torch.cat([self.We(X_nucl), self.Wpbs(X_pbs), self.Wrt(X_rt)], axis=-1)


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
        self.neg_inf = -1e6

    def forward(self, X: torch.Tensor, mask=None):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''
        # scale the input and query vector
        # scaling by the forth root of the input dimension
        # this is to prevent the dot product from becoming too large
        # print('X.shape', X.shape)
        X_scaled = X / torch.tensor(self.input_dim ** (1/4), device=X.device)
        queryv_scaled = self.queryv / torch.tensor(self.input_dim ** (1/4), device=self.queryv.device)
        
        # using matmul to compute tensor vector multiplication
        # produce attention weights of size (bsize, seqlen)
        attn_w = X_scaled.matmul(queryv_scaled)
        # print('attn_w.shape', attn_w.shape)

        # apply mask if available
        if mask is not None:
            # mask is of same size with attn_w
            # (batch, seqlen)
            # fill with neginf where mask == 0  
            attn_w = attn_w.masked_fill(mask == 0, self.neg_inf)
            # print('attn_w masked:\n', attn_w)

        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        # print('attn_w_normalized masked:\n', attn_w_normalized)
        
        if mask is not None:
            # ensures that the attention weights are 0 where the mask is 0
            # guarantees that they have no contribution to the final output
            attn_w_normalized = attn_w_normalized * mask


        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_w_normalized.unsqueeze(1).bmm(X).squeeze(1)
        
        del X_scaled, queryv_scaled, attn_w
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_w_normalized

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
        
        self.We = nn.Linear(input_dim, embed_dim, bias=True, dtype=torch.float32)
        # create a pipeline of MLP blocks
        encunit_layers = [MLPBlock(embed_dim,
                                   embed_dim,
                                   mlp_embed_factor,
                                   nonlin_func, 
                                   pdropout)
                          for i in range(num_encoder_units)]

        # create a sequential model from the MLP blocks
        self.encunit_pipeline = nn.Sequential(*encunit_layers)

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """
        X = X.to(torch.float32)
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

        # output embedding, returns the mean of the distribution
        self.W_mu = nn.Linear(embed_dim, outp_dim)
        self.softplus = nn.Softplus()
        
        # # output distribution
        # self.W_sigma = nn.Linear(embed_dim, outp_dim)
        # self.solfmax = nn.Softmax(dim=1)

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)

        mu = self.W_mu(out)
        return self.softplus(mu)

class Pridict(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 z_dim=32, 
                 device='cuda',
                 num_hiddenlayers=2, 
                 bidirection= True, 
                 dropout=0.15, 
                 rnn_class=nn.modules.rnn.GRU, 
                 nonlinear_func=nn.ReLU(),
                 fdtype = torch.float32,
                #   annot_embed=8,
                #   embed_dim=128,
                 annot_embed=2,
                 embed_dim = 4,
                 feature_dim=24,
                 mlp_embed_factor=2,
                 num_encoder_units=2,
                 num_hidden_layers=2,
                 assemb_opt='stack',
                 sequence_length=99):
        
        super().__init__()
        self.fdtype = fdtype
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_hiddenlayers = num_hiddenlayers
        self.device = device
        self.rnninput_dim = self.input_dim

        self.init_annot_embed = AnnotEmbeder_WTSeq(embed_dim=embed_dim,
                                                annot_embed_dim=annot_embed,
                                                assemb_opt=assemb_opt)
        self.mut_annot_embed = AnnotEmbeder_MutSeq(embed_dim=embed_dim,
                                              annot_embed_dim=annot_embed,
                                              assemb_opt=assemb_opt)
        if assemb_opt == 'stack':
            init_embed_dim = embed_dim + 3*annot_embed
            mut_embed_dim = embed_dim + 2*annot_embed
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2
        else:
            init_embed_dim = embed_dim
            mut_embed_dim = embed_dim
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2 
            
        self.sequence_length = sequence_length

        # encoder 1
        self.wt_encoder = RNN_Net(input_dim =init_embed_dim,
                              hidden_dim=embed_dim,
                              z_dim=z_dim,
                              device=device,
                              num_hiddenlayers=num_hidden_layers,
                              bidirection=bidirection,
                              rnn_pdropout=dropout,
                              rnn_class=rnn_class,
                              nonlinear_func=nonlinear_func,
                              fdtype=fdtype)
        # encoder 2
        self.mut_encoder= RNN_Net(input_dim =mut_embed_dim,
                              hidden_dim=embed_dim,
                              z_dim=z_dim,
                              device=device,
                              num_hiddenlayers=num_hidden_layers,
                              bidirection=bidirection,
                              rnn_pdropout=dropout,
                              rnn_class=rnn_class,
                              nonlinear_func=nonlinear_func,
                              fdtype=fdtype)

        self.local_featemb_wt_attn = FeatureEmbAttention(z_dim)
        self.local_featemb_mut_attn = FeatureEmbAttention(z_dim)

        self.global_featemb_wt_attn = FeatureEmbAttention(z_dim)
        self.global_featemb_mut_attn = FeatureEmbAttention(z_dim)

        # encoder 3
        # self.seqlevel_featembeder = MLPEmbedder(input_dim=feature_dim,
        #                                    embed_dim=z_dim,
        #                                    mlp_embed_factor=1,
        #                                    nonlin_func=nonlinear_func,
        #                                    pdropout=dropout, 
        #                                    num_encoder_units=1)
        self.seqlevel_featembeder = self.d = nn.Sequential(
                        nn.Linear(24, 96, bias=False),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(96, 64, bias=False),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 128, bias=False)
                    )

        # decoder
        # self.decoder  = MLPDecoder(5*z_dim,
        #                       embed_dim=z_dim,
        #                       outp_dim=1, # output is a scalar
        #                       mlp_embed_factor=2,
        #                       nonlin_func=nonlinear_func, 
        #                       pdropout=dropout, 
        #                       num_encoder_units=1)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(16 + 128),
            nn.Dropout(dropout),
            nn.Linear(16 + 128, 1, bias=True),
        )
        
    # skorch should be able to handle batching
    def forward(self, X_nucl, X_proto, X_pbs, X_rt, X_mut_nucl, X_mut_pbs, X_mut_rt, features):
        """
        Args:
            X_nucl: tensor, (batch, seqlen) representing nucleotide sequence
            X_proto: tensor, (batch, seqlen) representing location of protospacer sequence, dtype=torch.bool
            X_pbs: tensor, (batch, seqlen) representing location of PBS sequence, dtype=torch.bool
            X_rt: tensor, (batch, seqlen) representing location of RTT sequence, dtype=torch.bool
            X_mut_nucl: tensor, (batch, seqlen) representing mutated nucleotide sequence
            X_mut_pbs: tensor, (batch, seqlen) representing location of mutated PBS sequence, dtype=torch.bool
            X_mut_rt: tensor, (batch, seqlen) representing location of mutated RTT sequence, dtype=torch.bool
            features: tensor, (batch, feature_dim) representing feature vector
        """
        # process feature embeddings
        wt_embed = self.init_annot_embed(X_nucl, X_proto, X_pbs, X_rt)
        mut_embed = self.mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)
                        
        # rnn encoding
        # sequence lengths record the true length of the sequences without padding
        sequence_lengths = torch.sum(X_nucl != -1, axis=1)
        _, z_wt = self.wt_encoder(wt_embed, sequence_lengths)
        _, z_mut = self.mut_encoder(mut_embed, sequence_lengths)
        
        # print('z_wt', z_wt.shape)
        
        # attention mechanism
        # global attention
        # mask out regions that are part of the padding
        # mask is 1 where the padding is not present
        # print(X_nucl.shape)
        wt_mask = MaskGenerator.create_content_mask(X_nucl.shape, sequence_lengths)
        mut_mask = MaskGenerator.create_content_mask(X_mut_nucl.shape, sequence_lengths)
          
        # mask out the regions not part of the rtt using the X_rt tensor
        # X_rt is 0 where the RTT is not present
        # mask is 1 where the RTT is present
        wt_mask_local = X_rt
        mut_mask_local = X_mut_rt
        
        # replace 2 with 0
        # wt_mask_local[wt_mask_local == 2] = 0
        # mut_mask_local[mut_mask_local == 2] = 0
        
        # move the masks to the device
        wt_mask = wt_mask.to(self.device)
        mut_mask = mut_mask.to(self.device)
        wt_mask_local = wt_mask_local.to(self.device)
        mut_mask_local = mut_mask_local.to(self.device)
        
        local_attention_wt, _ = self.local_featemb_wt_attn(z_wt, wt_mask_local)
        local_attention_mut, _ = self.local_featemb_mut_attn(z_mut, mut_mask_local)
        
        global_attention_wt, _ = self.global_featemb_wt_attn(z_wt, wt_mask)
        global_attention_mut, _ = self.global_featemb_mut_attn(z_mut, mut_mask)
        
        # MLP feature embedding
        features_embed = self.seqlevel_featembeder(features)
        
        # concatenate the features
        z = torch.cat([local_attention_wt, local_attention_mut, global_attention_wt, global_attention_mut, features_embed], axis=1)
                
        # decoder
        mu_logit = self.decoder(z)
        # mu = torch.exp(mu_logit)
        
        del wt_embed, mut_embed, z_wt, z_mut, wt_mask, mut_mask, wt_mask_local, mut_mask_local, local_attention_wt, local_attention_mut, global_attention_wt, global_attention_mut, features_embed
        
        return torch.functional.F.softplus(mu_logit)
        
def preprocess_pridict(X_train: pd.DataFrame) -> Dict[str, torch.Tensor]:
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
        
    protospacer_location = X_train.loc[:, 'protospacer-location-l'].values
    
    pbs_start = X_train.loc[:, 'pbs-location-l'].values
    rtt_start = X_train.loc[:, 'rtt-location-l'].values
    
    
    mut_type = X_train.loc[:, 'mut-type'].values
    
    edit_length = X_train.loc[:, 'edit-length'].values
    pbs_length = X_train.loc[:, 'pbs-length'].values
    # rtt length wt
    rtt_length = X_train.loc[:, 'rtt-location-r'].values - rtt_start
    # rtt length mut
    rtt_length_mut = X_train.loc[:, 'rtt-location-r'].values - rtt_start

        
    X_pbs = torch.zeros((len(wt_seq), len(wt_seq[0])))    
    X_rtt = torch.zeros((len(wt_seq), len(wt_seq[0])))    
    X_proto = torch.zeros((len(wt_seq), len(wt_seq[0])))
    X_rtt_mut = torch.zeros((len(wt_seq), len(wt_seq[0])))
    
    
    for i in range(len(wt_seq)):
        # X_pbs[i, pbs_start[i]:pbs_start[i]+pbs_length[i]] = 1
        # X_rtt[i, rtt_start[i]:rtt_start[i]+rtt_length[i]] = 1
        # X_rtt_mut[i, rtt_start[i]:rtt_start[i]+rtt_length_mut[i]] = 1
        # X_proto[i, protospacer_location[i]:protospacer_location[i]+20] = 1
        for j in range(int(pbs_start[i]), int(pbs_start[i]+pbs_length[i])):
            X_pbs[i, j] = 1
        for j in range(int(rtt_start[i]), int(rtt_start[i]+rtt_length[i])):
            X_rtt[i, j] = 1
        for j in range(int(rtt_start[i]), int(rtt_start[i]+rtt_length_mut[i])):
            X_rtt_mut[i, j] = 1
        for j in range(int(protospacer_location[i]), int(protospacer_location[i]) + 20):
            X_proto[i, j] = 1
        
        # # annotate the padding regions
        # for j in range(len(wt_seq[i])):
        #     if wt_seq[i][j] == 'N':
        #         X_pbs[i, j] = 2
        #         X_rtt[i, j] = 2
        #         X_proto[i, j] = 2
        #         X_rtt_mut[i, j] = 2
        # print(f'X_pbs: {X_pbs[i]}')
        # print(f'X_rtt: {X_rtt[i]}')
        # print(f'X_proto: {X_proto[i]}')
        # print(f'X_rtt_mut: {X_rtt_mut[i]}')
                            
    nut_to_ix = {'N': 4, 'A': 0, 'T': 1, 'C': 2, 'G': 3}
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
        'features': torch.tensor(features).float()
    }
    
    return result
    
    
def train_pridict(train_fname: str, lr: float, batch_size: int, epochs: int, patience: int, num_runs: int, num_features: int, dropout: float = 0.1) -> skorch.NeuralNetRegressor:
    """train the pridict model

    Args:
        train_fname (str): _description_

    Returns:
        Pridict: _description_
    """
    # load a dp dataset
    dp_dataset = pd.read_csv(os.path.join('models', 'data', 'pridict', train_fname))
    
    # remove rows with nan values
    dp_dataset = dp_dataset.dropna()
    
    # TODO read the top 2000 rows only during development
    # dp_dataset = dp_dataset.head(2000)
    
    # standardize the scalar values at column 7:26
    # scalar = StandardScaler()
    # dp_dataset.iloc[:, 13:26] = scalar.fit_transform(dp_dataset.iloc[:, 13:26])
    
    # data origin
    data_origin = os.path.basename(train_fname).split('-')[1]
    
    sequence_length = len(dp_dataset['wt-sequence'].values[0])
    
    fold = 5
    
    # device
    device = torch.device('cuda')
    
    for i in range(fold):
        print(f'Fold {i+1} of {fold}')
        
        # if os.path.isfile(os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-best.pt")):
        #     continue
        
        train = dp_dataset[dp_dataset['fold']!=i]
        X_train = train
        print(X_train.columns)
        y_train = train.iloc[:, -2]
        
        # if adjustment == 'log':
        #     y_train = np.log1p(y_train)

        X_train = preprocess_pridict(X_train)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        print("Training PRIDICT model...")
        
        best_val_loss = np.inf
    
        for j in range(0, num_runs):
            print(f'Run {j+1} of {num_runs}')
            # model
            m = Pridict(input_dim=5,hidden_dim=32, sequence_length=sequence_length, dropout=dropout)
            
            # skorch model
            model = skorch.NeuralNetRegressor(
                m,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                optimizer__weight_decay=1e-5,
                device=device,
                batch_size=batch_size,
                max_epochs=epochs,
                train_split= skorch.dataset.ValidSplit(cv=5),
                # early stopping
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=patience),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', 
                                    f_params=os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), 
                                    f_optimizer=None, 
                                    f_history=None,
                                    f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1, eta_min=1e-6),   
                    # skorch.callbacks.ProgressBar()
                ]
            )
            
            model.initialize()
            
            model.fit(X_train, y_train)
            
            if np.min(model.history[:, 'valid_loss']) < best_val_loss:
                print(f'Best validation loss: {np.min(model.history[:, "valid_loss"])}')
                best_val_loss = np.min(model.history[:, 'valid_loss'])
                # rename the model file to the best model
                os.rename(os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
            else: # delete the last model
                print(f'Validation loss: {np.min(model.history[:, "valid_loss"])} is not better than {best_val_loss}')
                os.remove(os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"))
                
            del model, m
            torch.cuda.empty_cache()

        
    # return model

def predict(test_fname: str, num_features: int=24, device: str = 'cuda', dropout: float=0) -> skorch.NeuralNetRegressor:
    # model name
    fname = os.path.basename(test_fname)
    model_name =  fname.split('.')[0]
    data_source = model_name.split('-')[1:]
    data_source = '-'.join(data_source)
    model_name = '-'.join(model_name.split('-')[1:])
    models = [os.path.join('models', 'trained-models', 'pridict', f'pd-{data_source}-fold-{i}.pt') for i in range(1, 6)]
    # Load the data
    test_data_all = pd.read_csv(os.path.join('models', 'data', 'pridict', f'pd-{data_source}.csv'))    
    # remove rows with nan values
    test_data_all = test_data_all.dropna()
    # transform to float
    test_data_all.iloc[:, 2:26] = test_data_all.iloc[:, 2:26].astype(float)

    sequence_length = len(test_data_all['wt-sequence'].values[0])

    m = Pridict(input_dim=5,hidden_dim=32, sequence_length=sequence_length, dropout=dropout)
    
    pd_model = skorch.NeuralNetRegressor(
        m,
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        device=device,
    )

    prediction = dict()
    performance = []
    
    fold = 5
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Predicting PRIDICT model...')
    print(f'Using device: {device}')

    # Load the models
    for i, model in enumerate(models):
        test_data = test_data_all[test_data_all['fold']==i]
        X_test = test_data
        y_test = test_data.iloc[:, -2]
        X_test = preprocess_pridict(X_test)
        y_test = y_test.values
        pd_model.initialize()
        # if adjustment:
        #     pd_model.load_params(f_params=os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"), f_optimizer=os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"), f_history=os.path.join('models', 'trained-models', 'pridict', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json"))
        # else:
        pd_model.load_params(f_params=model)
        print(f'Predicting fold {i+1}...')
        
        y_pred = pd_model.predict(X_test).flatten()
        # if adjustment == 'log':
        #     y_pred = np.expm1(y_pred)

        print(f'Fold {i + 1} RMSE: {np.sqrt(np.mean((y_test - y_pred)**2))}')

        pearson = np.corrcoef(y_test, y_pred)[0, 1]
        spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

        print(f'Fold {i + 1} Pearson: {pearson}, Spearman: {spearman}')

        prediction[i] = y_pred
        performance.append((pearson, spearman))
    
    del pd_model, m
    torch.cuda.empty_cache()

    return prediction


def fine_tune_pridict(fine_tune_fname: str=None):    
    # load the fine tune datasets
    if not fine_tune_fname:
        fine_tune_data = glob(os.path.join('models', 'data', 'pridict', '*small*.csv'))
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
            model = Pridict(input_dim=5,hidden_dim=32, sequence_length=sequence_length, dropout=0)
            model.load_state_dict(torch.load('models/trained-models/pridict/dp-hek293t-pe2-fold-1.pt', map_location=device))
            
            # freeze the layers other than head and feature mlps
            for param in model.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.d.parameters():
                param.requires_grad = True
                
            # skorch wrapper
            dp_model = skorch.NeuralNetRegressor(
                model,
                criterion=nn.MSELoss,
                optimizer=torch.optim.Adam,
                device=device,
                warm_start=True,
                optimizer__lr=0.005,
                max_epochs=200,
                batch_size=1024,
                train_split= skorch.dataset.ValidSplit(cv=5),
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=30),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'models/trained-models/pridict/pd-{data_source}-fold-{fold}.pt', f_optimizer=None, f_history=None, f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1),
                ]
            )
            
            y_fine_tune = fine_tune.iloc[:, -2]
            X_fine_tune = preprocess_pridict(fine_tune)
            y_fine_tune = y_fine_tune.values
            y_fine_tune = y_fine_tune.reshape(-1, 1)
            y_fine_tune = torch.tensor(y_fine_tune, dtype=torch.float32)
            
            # train the model
            dp_model.fit(X_fine_tune, y_fine_tune)


def pridict(save_path: str, fine_tune: bool = False) -> skorch.NeuralNet:
    '''
    Returns the PRIDICT model wrapped by skorch
    '''
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = Pridict(input_dim=5,hidden_dim=32, dropout=0)
    if fine_tune:
        # load the dp hek293t pe 2 model
        m.load_state_dict(torch.load('models/trained-models/pridict/dp-hek293t-pe2-fold-1.pt', map_location=device))
        
        # freeze the layers other than head and feature mlps
        for param in m.parameters():
            param.requires_grad = False
        for param in m.head.parameters():
            param.requires_grad = True
        for param in m.d.parameters():
            param.requires_grad = True
            
    model = skorch.NeuralNetRegressor(
        m,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        device=device,
        batch_size=1024,
        max_epochs=500,
        optimizer__lr=0.0025 if not fine_tune else 0.001,
        train_split= skorch.dataset.ValidSplit(cv=5),
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=20),
            skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'{save_path}.pt', f_optimizer=None, f_history=None, f_criterion=None),
            skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1),
        ]
    )
    
    return model

class WeightedSkorch(skorch.NeuralNet):
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        # make sure to set reduce=False in your criterion, since we need the loss
        # for each sample so that it can be weighted
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        # override get_loss to use the sample_weight from X
        loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
        sample_weight = skorch.utils.to_tensor(X['sample_weight'], device=self.device)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

def pridict_weighted(save_path: str, fine_tune: bool=False) -> skorch.NeuralNet:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = Pridict(input_dim=5,hidden_dim=32, dropout=0)
    if fine_tune:
        # load the dp hek293t pe 2 model
        m.load_state_dict(torch.load('models/trained-models/pridict/dp-hek293t-pe2-fold-1.pt', map_location=device))
        
        # freeze the layers other than head and feature mlps
        for param in m.parameters():
            param.requires_grad = False
        for param in m.head.parameters():
            param.requires_grad = True
        for param in m.d.parameters():
            param.requires_grad = True
            
    model = WeightedSkorch(
        m,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        batch_size=1024,
        max_epochs=500,
        optimizer__lr=0.0025,
        train_split= skorch.dataset.ValidSplit(cv=5),
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=20),
            skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'{save_path}.pt', f_optimizer=None, f_history=None, f_criterion=None),
            skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1),
        ]
    )
    
    return model