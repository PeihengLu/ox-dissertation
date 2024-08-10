from torch import nn
import torch
import pandas as pd
import numpy as np
import os
import skorch
import scipy
import math

import sys
sys.path.append('../')
from typing import Dict

from utils.ml_utils import clones
from flash_attn import flash_attn_qkvpacked_func

class SequenceEmbedder(nn.Module):
    def __init__(self, embed_dim, sequence_length=99):
        super().__init__()
        self.num_nucl = 4 # nucleotide embeddings
        self.num_inidc = 2 # padding index for protospacer, PBS and RTT
        # wt+mut sequence embedding
        self.We = nn.Embedding(self.num_nucl+1, embed_dim, padding_idx=0)
        # # protospacer embedding
        # self.Wproto = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # # PBS embedding
        # self.Wpbs = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        # # RTT embedding
        # self.Wrt = nn.Embedding(self.num_inidc+1, annot_embed, padding_idx=self.num_inidc)
        
        # Create a matrix of shape (max_len, d_model) for position encodings
        position_encoding = torch.zeros(sequence_length, embed_dim)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra batch dimension to the position encoding
        position_encoding = position_encoding.unsqueeze(0)
        
        # Register the position encoding as a buffer, which is a tensor not considered a model parameter
        self.register_buffer('position_encoding', position_encoding)
    
    def forward(self, X_nucl, padding_mask=None):
        # if self.assemb_opt == 'add':
        #     return self.We(X_nucl) + self.Wproto(X_proto) + self.Wpbs(X_pbs) + self.Wrt(X_rt)
        # elif self.assemb_opt == 'stack':
        #     return torch.cat([self.We(X_nucl), self.Wproto(X_proto), self.Wpbs(X_pbs), self.Wrt(X_rt)], axis=-1)
        x = self.We(X_nucl)
        # position embedding for non padding sequence using sinusoidal function
        x = x + self.position_encoding[:, :x.size(1)]
        
        if padding_mask is not None:
            # Expand the mask to match the dimensions of x (seq_len, batch_size, d_model)
            padding_mask = padding_mask.transpose(0, 1).unsqueeze(2).expand_as(x)
            x = x.masked_fill(padding_mask, 0)
        
        return x


# feature processing
class MLPEmbedder(nn.Module):
    def __init__(self,
                 num_features, # number of features
                 embed_dim, # number of features after embedding
                 mlp_embed_factor=2,
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=2):
        
        super().__init__()
        
        self.We = nn.Linear(num_features, embed_dim, bias=True)
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
        
        self.softplus = nn.Softplus()

    def forward(self, X):
        """
        Args:
            X: tensor, float32, (batch, embed_dim) representing x_target
        """

        X = self.We(X)
        out = self.encunit_pipeline(X)

        mu = self.W_mu(out)
        
        return self.softplus(mu)


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
    
class Projection(nn.Module):
    """projecction of the input to the query, key and value spaces
    """
    def __init__(self, input_dim, num_heads):
        super(Projection, self).__init__()
        # input dim is the sequence length
        # drop out must be 0 during inference
        # projection of the input to the query, key and value spaces
        self.Wqs = clones(nn.Linear(input_dim, input_dim, bias=False), num_heads)
        self.Wks = clones(nn.Linear(input_dim, input_dim, bias=False), num_heads)
        self.Wvs = clones(nn.Linear(input_dim, input_dim, bias=False), num_heads)

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: tensor, float32, (batch, sequence length, embed_dim) representing the input
        """
        # project the input to the query, key and value spaces
        # the output is a list of tensors
        Qs = [Wq(X) for Wq in self.Wqs]
        Ks = [Wk(X) for Wk in self.Wks]
        Vs = [Wv(X) for Wv in self.Wvs]
        
        return Qs, Ks, Vs


class TransformerBlock(nn.Module):
    """
    Transformer = FlashAttention + ResidualConnection + LayerNorm + FeedForward + ResidualConnection + LayerNorm
    """
    def __init__(self, attn_dim, pdropout, mlp_embed_factor, nonlin_func):
        super(TransformerBlock, self).__init__()
        self.pdropout = pdropout
        self.layer_norm_1 = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(pdropout)
        # position wise feed forward
        self.position_feed_forward = nn.Sequential(
            nn.Linear(attn_dim, attn_dim*mlp_embed_factor),
            nonlin_func,
            nn.Linear(attn_dim*mlp_embed_factor, attn_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(attn_dim)
        
    def forward(self, Qs, Ks, Vs, X) -> torch.Tensor:
        """forward pass of the transformer block

        Args:
            Qs (torch.Tensor): tensor, float32, (batch, sequence length, number of heads, attn_dim) representing the query
            Ks (torch.Tensor): tensor, float32, (batch, sequence length, number of heads, attn_dim) representing the key
            Vs (torch.Tensor): tensor, float32, (batch, sequence length, number of heads, attn_dim) representing the value
            X (torch.Tensor): tensor, float32, (batch, sequence length, attn_dim) representing the input used in the residual connection

        Returns:
            torch.Tensor: tensor, float32, (batch, sequence length, attn_dim) representing the output of the transformer block
        """
        # apply the attention mechanism
        # attn_dim is the embedding dimension of the input
        qkv = torch.stack([Qs, Ks, Vs], dim=-1) # (batch, sequence length, num_heads, attn_dim, 3)
        # transform into shape (batch, sequence length, 3, num_heads, attn_dim)
        qkv = qkv.permute(0, 1, 4, 2, 3)
        
        attn = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.pdropout)
        
        # add and norm
        # apply the residual connection
        attn = attn + X
        attn = self.dropout(attn)
        attn = self.layer_norm_1(attn)
        
        # position wise feed forward
        attn = self.position_feed_forward(attn)
        
        # add and norm
        # apply the residual connection
        attn = attn + X
        attn = self.dropout(attn)
        attn = self.layer_norm_2(attn)
        
        return attn
    
class TransformerDecoderInput(nn.Module):
    def __init__(self, input_dim, num_heads, attn_dim, pdropout):
        super(TransformerDecoderInput, self).__init__()
        self.dropout = pdropout
        self.proj_layer = Projection(input_dim, num_heads)
        
    def forward(self, Qs: torch.Tensor, Ks: torch.Tensor, Vs: torch.Tensor) -> torch.Tensor:
        """forward pass of the transformer decoder input

        Args:
            Qs (torch.Tensor): query tensor, float32, (batch, sequence length, input_dim)
            Ks (torch.Tensor): key tensor, float32, (batch, sequence length, input_dim)
            Vs (torch.Tensor): value tensor, float32, (batch, sequence length, input_dim)

        Returns:
            torch.Tensor: attention output tensor, float32, (batch, sequence length, attn_dim)
        """
        qkv = torch.stack([Qs, Ks, Vs], dim=-1) # (batch, sequence length, num_heads, attn_dim, 3)
        # transform into shape (batch, sequence length, 3, num_heads, attn_dim)
        qkv = qkv.permute(0, 1, 4, 2, 3)
        # returns attention in shape (batch, sequence length, number of heads, attn_dim)
        attn = flash_attn_qkvpacked_func(qkv=qkv, dropout_p=self.dropout)
        return attn


class Transformer(nn.Module):
    def __init__(self, sequence_len, embed_dim, num_heads, pdropout, mlp_embed_factor, nonlin_func, num_encoder_units):
        super(Transformer, self).__init__()
        self.sequence_len = sequence_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dim = embed_dim 
        self.pdropout = pdropout
        self.mlp_embed_factor = mlp_embed_factor
        self.nonlin_func = nonlin_func
        self.num_encoder_units = num_encoder_units
        
        # projects the wild type sequence to the query, key and value spaces
        self.proj_layer_wt = Projection(sequence_len, num_heads)
        
        # transformer blocks
        # the encoder transformer block
        self.encoder_transformer = TransformerBlock(embed_dim, pdropout, mlp_embed_factor, nonlin_func)
        
        # the decoder transformer blocks
        self.decoder_input = TransformerDecoderInput(sequence_len, num_heads, embed_dim, pdropout)
        self.decoder_transformer = TransformerBlock(embed_dim, pdropout, mlp_embed_factor, nonlin_func)

    def forward(self, X_wt: torch.Tensor, X_mut: torch.Tensor) -> torch.Tensor:
        """forward pass of the transformer

        Args:
            X_wt (torch.Tensor): tensor, float32, (batch, sequence length, input_dim) representing the wild type sequence
            X_mut (torch.Tensor): tensor, float32, (batch, sequence length, input_dim) representing the mutant type sequence

        Returns:
            torch.Tensor: tensor, float32, (batch, sequence length, input_dim) representing the output of the transformer
        """
        # project the wild type and mutant sequences to the query, key and value spaces
        Qs_wt, Ks_wt, Vs_wt = self.proj_layer_wt(X_wt)
        Qs_mut, Ks_mut, Vs_mut = self.proj_layer_wt(X_mut)
        
        # apply the encoder transformer block
        wt_attn = self.encoder_transformer(Qs_wt, Ks_wt, Vs_wt)
        # apply the decoder transformer block
        mut_key = self.decoder_input(Qs_mut, Ks_mut, Vs_mut)
        
        # use the output of the mutated sequence as the query
        cross_attn = self.decoder_transformer(mut_key, wt_attn, wt_attn, X_mut)
        
        return cross_attn


class PrimeDesignTransformer(nn.Module):
    def __init__(self, embed_dim, sequence_length, num_features, num_heads, pdropout, mlp_embed_factor, nonlin_func, num_encoder_units):
        super(PrimeDesignTransformer, self).__init__()
        self.sequence_embedder = SequenceEmbedder(embed_dim, sequence_length)
        self.mlp_embedder = MLPEmbedder(num_features, embed_dim, mlp_embed_factor, nonlin_func, pdropout, num_encoder_units)
        self.transformer = Transformer(sequence_length, embed_dim, num_heads, pdropout, mlp_embed_factor, nonlin_func, num_encoder_units)
        self.mlp_decoder = MLPDecoder(embed_dim, embed_dim, 1, mlp_embed_factor, nonlin_func, pdropout, num_encoder_units)
        
    def forward(self, X_nucl: torch.Tensor, X_mut_nucl: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """forward pass of the PrimeDesignTransformer

        Args:
            X_nucl (torch.Tensor): embedding of the wild type sequence
            X_mut_nucl (torch.Tensor): _description_
            features (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # create padding mask for 0 values
        padding_mask = (X_nucl == 0)
    
        # sequence embedding
        wt_seq = self.sequence_embedder(X_nucl, padding_mask)
        mut_seq = self.sequence_embedder(X_mut_nucl, padding_mask)
        
        # sequence transformer
        seq_out = self.transformer(wt_seq, mut_seq)
        
        # feature embedding
        features = self.mlp_embedder(features)
        
        # concatenate the sequence and feature embeddings
        out = torch.cat([seq_out, features], axis=-1)
        
        # decoder output, solfplus activation function is used to ensure the output is positive
        prediction = self.mlp_decoder(out)
        
        return prediction


def preprocess_transformer(X_train: pd.DataFrame) -> Dict[str, torch.Tensor]:
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
        
    nut_to_ix = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    X_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in wt_seq])
    X_mut_nucl = torch.tensor([[nut_to_ix[n] for n in seq] for seq in mut_seq])
    
    result = {
        'X_nucl': X_nucl,
        'X_mut_nucl': X_mut_nucl,
        'features': torch.tensor(features)
    }
    
    return result

def train_transformer(train_fname: str, lr: float, batch_size: int, epochs: int, patience: int, num_runs: int, num_features: int, dropout: float = 0.1, percentage: str = 1) -> skorch.NeuralNetRegressor:
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
    
    for i in range(fold):
        print(f'Fold {i+1} of {fold}')
        
        train = dp_dataset[dp_dataset['fold']!=i]
        X_train = train.iloc[:, :num_features+2]
        print(X_train.columns)
        y_train = train.iloc[:, -2]

        X_train = preprocess_transformer(X_train)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        print("Training Transformer model...")
        
        best_val_loss = np.inf
    
        for j in range(num_runs):
            print(f'Run {j+1} of {num_runs}')
            # model
            m = PrimeDesignTransformer(embed_dim=5, sequence_length=99, num_heads=2,pdropout=dropout, mlp_embed_factor=2, nonlin_func=nn.ReLU(), num_encoder_units=2, num_features=num_features)
            
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
                                    f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), 
                                    f_optimizer=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"), 
                                    f_history=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"),
                                    f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1, eta_min=1e-3),
                    # skorch.callbacks.ProgressBar()
                ]
            )
            
            model.initialize()
            
            model.fit(X_train, y_train)
            
            if np.min(model.history[:, 'valid_loss']) < best_val_loss:
                print(f'Best validation loss: {np.min(model.history[:, "valid_loss"])}')
                best_val_loss = np.min(model.history[:, 'valid_loss'])
                # rename the model file to the best model
                os.rename(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
                os.rename(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"), os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"))
                os.rename(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"), os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json")) 
            else: # delete the last model
                print(f'Validation loss: {np.min(model.history[:, "valid_loss"])} is not better than {best_val_loss}')
                os.remove(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"))
                os.remove(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"))
                os.remove(os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"))       
            
        
    return model

def predict_transformer(test_fname: str, num_features: int, adjustment: str = None, device: str = 'cuda', dropout: float=0) -> skorch.NeuralNetRegressor:
    # model name
    fname = os.path.basename(test_fname)
    model_name =  fname.split('.')[0]
    model_name = '-'.join(model_name.split('-')[1:])
    models = [os.path.join('models', 'trained-models', 'transformer', f'{model_name}-fold-{i}.pt') for i in range(1, 6)]
    # Load the data
    test_data_all = pd.read_csv(os.path.join('models', 'data', 'transformer', test_fname))    
    # remove rows with nan values
    test_data_all = test_data_all.dropna()
    # transform to float
    test_data_all.iloc[:, 2:26] = test_data_all.iloc[:, 2:26].astype(float)

    m = PrimeDesignTransformer(embed_dim=5, sequence_length=99, num_heads=2, attn_dim=5, pdropout=dropout, mlp_embed_factor=2, nonlin_func=nn.ReLU(), num_encoder_units=2)
    
    pd_model = skorch.NeuralNetRegressor(
        m,
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        device=device,
    )

    prediction = {}
    performance = []
    
    fold = 5
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the models
    for i, model in enumerate(models):
        test_data = test_data_all[test_data_all['fold']==i]
        X_test = test_data.iloc[:, :num_features+2]
        y_test = test_data.iloc[:, -2]
        X_test = preprocess_transformer(X_test)
        y_test = y_test.values
        y_test = y_test.reshape(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        pd_model.initialize()
        if adjustment:
            pd_model.load_params(f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"), f_optimizer=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"), f_history=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json"))
        else:
            pd_model.load_params(f_params=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"), f_optimizer=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"), f_history=os.path.join('models', 'trained-models', 'transformer', f"{'-'.join(os.path.basename(test_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json"))
        
        y_pred = pd_model.predict(X_test)
        if adjustment == 'log':
            y_pred = np.expm1(y_pred)

        pearson = np.corrcoef(y_test.T, y_pred.T)[0, 1]
        spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

        print(f'Fold {i + 1} Pearson: {pearson}, Spearman: {spearman}')

        prediction[i] = y_pred
        performance.append((pearson, spearman))
    
    return prediction, performance