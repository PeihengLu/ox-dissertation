# PRIDICT Model by Gerald et al 2023
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from typing import Dict
import skorch

import os
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
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=0)
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
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=0)
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
        return mu
    
        # calculate the distribution
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
        self.seqlevel_featembeder = MLPEmbedder(input_dim=feature_dim,
                                           embed_dim=z_dim,
                                           mlp_embed_factor=mlp_embed_factor,
                                           nonlin_func=nonlinear_func,
                                           pdropout=dropout, 
                                           num_encoder_units=num_encoder_units)

        # decoder
        self.decoder  = MLPDecoder(5*z_dim,
                              embed_dim=z_dim,
                              outp_dim=1, # output is a scalar
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
        # process feature embeddings
        wt_embed = self.init_annot_embed(X_nucl, X_proto, X_pbs, X_rt)
        mut_embed = self.mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)
                
        # rnn encoding
        # sequence lengths record the true length of the sequences without padding
        sequence_lengths = torch.sum(X_nucl != 0, axis=1)
        _, z_wt = self.wt_encoder(wt_embed, sequence_lengths)
        _, z_mut = self.mut_encoder(mut_embed, sequence_lengths)
        

        # attention mechanism
        # global attention
        # mask out regions that are part of the padding
        # mask is 1 where the padding is not present
        wt_mask = MaskGenerator.create_content_mask(X_nucl.shape, sequence_lengths)
        mut_mask = MaskGenerator.create_content_mask(X_mut_nucl.shape, sequence_lengths)
        
        # mask out the regions not part of the rtt using the X_rt tensor
        # X_rt is 0 where the RTT is not present
        # mask is 1 where the RTT is present
        wt_mask_local = X_rt
        mut_mask_local = X_mut_rt
        
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
        mu = self.decoder(z)
        
        return mu
        
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
    
    
def train_pridict(train_fname: str, lr: float, batch_size: int, epochs: int, patience: int, num_runs: int, adjustment: str, num_features: int) -> skorch.NeuralNetRegressor:
    """train the pridict model

    Args:
        train_fname (str): _description_

    Returns:
        Pridict: _description_
    """
    # load a dp dataset
    dp_dataset = pd.read_csv(os.path.join('models', 'data', 'pridict', train_fname))
    
    # TODO read the top 2000 rows only during development
    # dp_dataset = dp_dataset.head(2000)
    
    # standardize the scalar values at column 2:26
    scalar = StandardScaler()
    dp_dataset.iloc[:, 2:26] = scalar.fit_transform(dp_dataset.iloc[:, 2:26])
    
    # data origin
    data_origin = os.path.basename(train_fname).split('-')[1]
    
    fold = 5
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(fold):
        print(f'Fold {i+1} of {fold}')
        train = dp_dataset[dp_dataset['fold']!=i]
        X_train = train.iloc[:, :num_features+2]
        y_train = train.iloc[:, -2]
        
        if adjustment == 'log':
            y_train = np.log1p(y_train)

        X_train = preprocess_pridict(X_train)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        print("Training DeepPrime model...")
        
        best_val_loss = np.inf
    
        for i in range(num_runs):
            # model
            m = Pridict(input_dim=5,
                        hidden_dim=32,
                        z_dim=16,
                        device=device,
                        num_hiddenlayers=1,
                        bidirection=False,
                        dropout=0.5,
                        rnn_class=nn.LSTM,
                        nonlinear_func=nn.ReLU(),
                        fdtype=torch.float32,
                        annot_embed=32,
                        embed_dim=64,
                        feature_dim=24,
                        mlp_embed_factor=2,
                        num_encoder_units=2,
                        num_hidden_layers=2,
                        assemb_opt='stack')
            
            # skorch model
            model = skorch.NeuralNetRegressor(
                m,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=lr,
                device=device,
                batch_size=batch_size,
                max_epochs=epochs,
                train_split= skorch.dataset.ValidSplit(cv=5),
                # early stopping
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=patience),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', 
                                    f_params=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"), 
                                    f_optimizer=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"), 
                                    f_history=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json"),
                                    f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=15, T_mult=1),
                    skorch.callbacks.ProgressBar()
                ]
            )
            
            model.fit(X_train, y_train)
        
        
    return model