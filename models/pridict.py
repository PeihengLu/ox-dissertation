# PRIDICT Model by Gerald et al 2023
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from typing import Dict

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
        self.device = device
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
  
        return(hidden, z_logit)
    
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
    def __init__(self, embed_dim, annot_embed, assemb_opt='add'):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 0 # padding index
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


# mutated sequence embedding
class AnnotEmbeder_MutSeq(nn.Module):
    def __init__(self, embed_dim, annot_embed, assemb_opt='add'):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 0 # padding index
        self.assemb_opt = assemb_opt
        # one hot encoding
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=0)
        # PBS embedding
        self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # RTT embedding
        self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
    
    def forward(self, X_nucl, X_pbs, X_rt):
        if self.assemb_opt == 'add':
            return self.We(X_nucl) + self.Wpbs(X_pbs) + self.Wrt(X_rt)
        elif self.assemb_opt == 'stack':
            return torch.cat([self.We(X_nucl), self.Wpbs(X_pbs), self.Wrt(X_rt)], axis=-1)


class SH_Attention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size, embed_size):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = embed_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
        self.neginf = -1e6
    
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length, sequence length) with 0/1 entries
                  (default None)
                  
        .. note:
            
            mask has to have at least one element in a row that is equal to one otherwise a uniform distribution
            will be genertaed when computing attn_w_normalized!
            
        """
        # print('---- SH layer ----')
        # print('Xin_q.shape', Xin_q.shape)
        # print('Xin_q.shape', Xin_k.shape)
        # print('Xin_v.shape', Xin_v.shape)

        # print('self.Wq:', self.Wq)
        # print('self.Wk:', self.Wk)
        # print('self.Wv:', self.Wv)

        X_q = self.Wq(Xin_q) # queries
        X_k = self.Wk(Xin_k) # keys
        X_v = self.Wv(Xin_v) # values
        
        # print('---- SH layer transform ----')
        # print('X_q.shape', X_q.shape)
        # print('X_k.shape', X_k.shape)
        # print('X_v.shape', X_v.shape)

        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        # (batch, sequence length, sequence length)
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # print('attn_w.shape:', attn_w.shape)
        # print()
         
        if mask is not None:
            # (batch, seqlen, seqlen)
            # if mask.dim() == 2: # assumption mask.shape = (seqlen, seqlen)
            #     mask = mask.unsqueeze(0) # add batch dimension
            # fill with neginf where mask == 0  
            attn_w = attn_w.masked_fill(mask == 0, self.neginf)
            # print('attn_w masked:\n', attn_w)

        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        # print('attn_w_normalized masked:\n', attn_w_normalized)
        
        if mask is not None:
            # for cases where the mask is all 0 in a row
            attn_w_normalized = attn_w_normalized * mask
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized


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

    def forward(self, X, mask=None):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_w = X_scaled.matmul(queryv_scaled)


        if mask is not None:
            # (batch, seqlen)
            # fill with neginf where mask == 0  
            attn_w = attn_w.masked_fill(mask == 0, self.neg_inf)
            # print('attn_w masked:\n', attn_w)

        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        # print('attn_w_normalized masked:\n', attn_w_normalized)
        
        if mask is not None:
            # for cases where the mask is all 0 in a row
            attn_w_normalized = attn_w_normalized * mask


        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_w_normalized.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_w_normalized

# feature processing
class MLPEmbedder(nn.Module):
    def __init__(self,
                 inp_dim,
                 embed_dim,
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

class Pridict(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 z_dim, 
                 device,
                 num_hiddenlayers=1, 
                 bidirection= False, 
                 dropout=0.5, 
                 rnn_class=nn.LSTM, 
                 nonlinear_func=nn.ReLU(),
                 fdtype = torch.float32,
                 annot_embed=32,
                 embed_dim=64,
                 feature_dim=24,
                 mlp_embed_factor=2,
                 num_encoder_units=2,
                 num_hidden_layers=2,
                 assemb_opt='add'):
        
        super().__init__()
        self.fdtype = fdtype
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_hiddenlayers = num_hiddenlayers
        self.device = device
        self.rnninput_dim = self.input_dim

        self.init_annot_embed = AnnotEmbeder_WTSeq(embed_dim=embed_dim,
                                                annot_embed=annot_embed,
                                                assemb_opt=assemb_opt)
        self.mut_annot_embed = AnnotEmbeder_MutSeq(embed_dim=embed_dim,
                                              annot_embed=annot_embed,
                                              assemb_opt=assemb_opt)
        if assemb_opt == 'stack':
            init_embed_dim = embed_dim + 3*annot_embed
            mut_embed_dim = embed_dim + 2*annot_embed
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2
        else:
            init_embed_dim = embed_dim
            mut_embed_dim = embed_dim
            z_dim = np.min([init_embed_dim, mut_embed_dim])//2 

        # encoder 1
        self.init_encoder = RNN_Net(input_dim =init_embed_dim,
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

        self.local_featemb_init_attn = FeatureEmbAttention(z_dim)
        self.local_featemb_mut_attn = FeatureEmbAttention(z_dim)

        self.global_featemb_init_attn = FeatureEmbAttention(z_dim)
        self.global_featemb_mut_attn = FeatureEmbAttention(z_dim)

        # encoder 3
        self.seqlevel_featembeder = MLPEmbedder(inp_dim=feature_dim,
                                           embed_dim=z_dim,
                                           mlp_embed_factor=mlp_embed_factor,
                                           nonlin_func=nonlinear_func,
                                           pdropout=dropout, 
                                           num_encoder_units=num_encoder_units)

        # decoder
        self.decoder  = MLPDecoder(5*z_dim,
                              embed_dim=z_dim,
                              outp_dim=1,
                              mlp_embed_factor=2,
                              nonlin_func=nonlinear_func, 
                              pdropout=dropout, 
                              num_encoder_units=1)
        
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
        wt_embed = self.init_annot_embed(X_nucl, X_proto, X_pbs, X_rt)
        mut_embed = self.mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)
        
        # feature embedding
        
        
        # attention mechanism
        
def preprocess_pridict(X_train: pd.DataFrame) -> Dict[str, torch.Tensor]:
    # sequence data
    wt_seq = X_train['wt-sequence']
    mut_seq = X_train['mut-sequence']
    # the rest are the features
    features = X_train.iloc[:, 2:26].values
    
    protospacer_location = X_train['protospacer-location-l']
    pbs_start = X_train['pbs-location-l-relative-protospacer'] + protospacer_location
    rtt_start = X_train['rtt-location-l-relative-protospacer'] + protospacer_location
    
    mut_type = X_train['mut-type']
    
    edit_length = X_train['edit-length']
    pbs_length = X_train['pbs-length']
    rtt_length = X_train['rtt-length']

    rtt_length_mut = []
    
    for i in range(len(wt_seq)):
        if mut_type[i] == 2:
            rtt_length_mut.append(rtt_length[i] - edit_length[i])
        elif mut_type[i] == 1:
            rtt_length_mut.append(rtt_length[i] + edit_length[i])
        else:
            rtt_length_mut.append(rtt_length[i])
        
    X_pbs = torch.zeros((len(wt_seq), len(wt_seq[0])))
    X_pbs[pbs_start:pbs_start+pbs_length] = 1
    
    X_rtt = torch.zeros((len(wt_seq), len(wt_seq[0])))
    X_rtt[rtt_start:rtt_start+rtt_length] = 1
    
    print(X_pbs.shape)
        
    nut_to_ix = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    

    
    