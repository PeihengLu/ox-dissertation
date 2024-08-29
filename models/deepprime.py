import os
import numpy as np
import pandas as pd
import sklearn.metrics
import skorch.helper
import torch
import torch.nn.functional as F
import torch.nn as nn
from glob import glob
from typing import Dict, Tuple
import sys
import scipy
import sklearn
import torch.utils
import skorch
from sklearn.preprocessing import StandardScaler
import logging
log = logging.getLogger(__name__)

util_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(util_path)

from utils.data_utils import onehot_encode_series
from utils.ml_utils import undersample


class DPEmbedding(nn.Module):
    '''
    Produce an embedding of the input nucleotide sequences
    '''
    def __init__(self):
        super(DPEmbedding, self).__init__()
        self.embedding = nn.Embedding(5, 4, padding_idx=0)
        
    def forward(self, g):
        return self.embedding(g)

class DeepPrime(nn.Module):
    '''
    requires hidden size and number of layers of the 
    GRU, number of features in the feature vector, and dropout rate
    '''
    def __init__(self, hidden_size, num_layers, num_features=24, dropout=0.1):
        super(DeepPrime, self).__init__()
        self.embedding = DPEmbedding()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=128, kernel_size=(2, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=108, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(108),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=108, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
        )

        self.r = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.s = nn.Linear(2 * hidden_size, 12, bias=False)

        self.d = nn.Sequential(
            nn.Linear(num_features, 96, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128, bias=False)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(140),
            nn.Dropout(dropout),
            nn.Linear(140, 1, bias=True),
        )

    # g is the stacked gene sequences(wildtype and edited) and x is the feature vector
    def forward(self, g, x, sample_weight=None):
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        g = g.to(device).long()
        x = x.to(device)        
        g = self.embedding(g)
        
        # Ensure g is 4D
        if g.dim() == 3:
            g = g.unsqueeze(3)  # Add an extra dimension at the end
        # print("Shape of g after unsqueeze:", g.shape)

        # reshape to the format (batch_size, channels, height, width)
        g = g.permute(0, 2, 1, 3)
        
        # Pass the data through the Conv2d layers
        g = self.c1(g)

        # Remove the last dimension (width=1)
        g = torch.squeeze(g, 3)

        # Pass the data through the Conv1d layers
        g = self.c2(g)

        # Transpose for the GRU layer
        g, _ = self.r(torch.transpose(g, 1, 2))

        # Get the last hidden state
        g = self.s(g[:, -1, :])

        x = self.d(x)
        # print("Shape of x after dense layers d:", x.shape)

        out = self.head(torch.cat((g, x), dim=1))
        # print("Shape of out after head:", out.shape)

        return F.softplus(out)
    
# Custom loss function, adjusting for more frequent low efficiency values
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.base_loss = nn.MSELoss()  # or nn.CrossEntropyLoss() for classification

    def forward(self, outputs, targets):
        weights = self.calculate_weights(targets)
        loss = self.base_loss(outputs, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()

    @staticmethod
    def calculate_weights(efficiencies):
        weights = torch.exp(6 * (torch.log(efficiencies + 1) - 3) + 1)
        weights = torch.min(weights, torch.tensor(5.0))
        return weights

# returns a loaded data loader
def preprocess_deep_prime(X_train: pd.DataFrame, source: str = 'dp', sample_weight = None) -> Dict[str, torch.Tensor]:
    '''
    Preprocesses the data for the DeepPrime model
    '''
    # sequence data
    wt_seq = X_train['wt-sequence'].values
    mut_seq = X_train['mut-sequence'].values

    # log.info(f'number of wildtype sequences: {len(wt_seq)}')
    
    # crop the sequences to 74bp if longer
    if len(wt_seq[0]) > 74:
        wt_seq = [seq[:74] for seq in wt_seq]
        mut_seq = [seq[:74] for seq in mut_seq]
    
    # the rest are the features
    features = X_train.iloc[:, 2:26].values
    features = features.astype(np.float32)

    # log.info(f'Features: {features[0]}')
    
    # concatenate the sequences
    seqs = []
    for wt, mut in zip(wt_seq, mut_seq):
        # log.info(f'Wildtype: {wt}, Mutant: {mut}')
        seqs.append(wt + mut)
    
    if source != 'org':
        nut_to_ix = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    else:
        nut_to_ix = {'x': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}

    # log.info(f'Sequences: {seqs}')

    output = {
        'g': torch.tensor([[nut_to_ix[n] for n in seq] for seq in seqs], dtype=torch.float32),
        'x': torch.tensor(features, dtype=torch.float32)
    }
    
    if sample_weight is not None:
        output['sample_weight'] = torch.tensor(sample_weight, dtype=torch.float32)
        
    return output

def train_deep_prime(train_fname: str, hidden_size: int, num_layers: int, num_features: int, dropout: float, device: str, epochs: int, lr: float, batch_size: int, patience: int, num_runs: int = 3, source: str = 'dp') -> skorch.NeuralNet:
    '''
    Trains the DeepPrime model
    '''
    # load a dp dataset
    if source == 'org': # dp features
        dp_dataset = pd.read_csv(os.path.join('models', 'data', 'deepprime-org', train_fname))
    else:
        dp_dataset = pd.read_csv(os.path.join('models', 'data', 'deepprime', train_fname))
    
    # standardize the scalar values at column 2:26
    # scalar = StandardScaler()
    # dp_dataset.iloc[:, 2:26] = scalar.fit_transform(dp_dataset.iloc[:, 2:26])
    
    # data origin
    data_origin = os.path.basename(train_fname).split('-')[1]
    
    fold = 5
    
    print(dp_dataset.columns)

    for i in range(fold):
        print(f'Fold {i+1} of {fold}')
        train = dp_dataset[dp_dataset['fold']!=i]
        X_train = train.iloc[:, :num_features+2]
        y_train = train.iloc[:, -2]


        X_train = preprocess_deep_prime(X_train, source)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        print("Training DeepPrime model...")
        
        best_val_loss = np.inf

        for j in range(num_runs):
            model = skorch.NeuralNetRegressor(
                DeepPrime(hidden_size, num_layers, num_features, dropout),
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
                                    f_params=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), 
                                    f_optimizer=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"), 
                                    f_history=os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"),
                                    f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=15, T_mult=1),
                    # skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau, monitor='valid_loss', factor=0.5, patience=3, min_lr=1e-6),
                    # skorch.callbacks.ProgressBar()
                ]
            )
            print(f'Run {j+1} of {num_runs}')
            # Train the model
            model.fit(X_train, y_train)
            # check if validation loss is better
            valid_losses = model.history[:, 'valid_loss']
            # find the minimum validation loss
            min_valid_loss = min(valid_losses)
            if min_valid_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {min_valid_loss}")
                best_val_loss = min_valid_loss
                # rename the save model 
                os.rename(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"), os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}.pt"))
                os.rename(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"), os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer.pt"))
                os.rename(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"), os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history.json"))
            else:
                print(f"Validation loss did not improve from {best_val_loss}")
                # remove the temporary files
                os.remove(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-tmp.pt"))
                os.remove(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-optimizer-tmp.pt"))
                os.remove(os.path.join('models', 'trained-models', 'deepprime', f"{'-'.join(os.path.basename(train_fname).split('.')[0].split('-')[1:])}-fold-{i+1}-history-tmp.json"))
        print("Training done.")
        
        del model
        torch.cuda.empty_cache()

    # return model

def predict(test_fname: str, hidden_size: int = 128, num_layers: int = 1, num_features: int = 24, dropout: float = 0, adjustment: str = None, source: str='dp') -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Make predictions using the DeepPrime model

    Args:
        test_fname (str): Base name of the test file
    Returns:
        Dict[str, np.ndarray]: The predictions result of the model from each fold
    """
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # model name
    fname = os.path.basename(test_fname)
    model_name =  fname.split('.')[0]
    data_source = model_name.split('-')[1:]
    data_source = '-'.join(data_source)
    model_name = '-'.join(model_name.split('-')[1:])
    models = [os.path.join('models', 'trained-models', 'deepprime', f'dp-{data_source}-fold-{i}.pt') for i in range(1, 6)]
    # Load the data
    test_data_all = pd.read_csv(os.path.join('models', 'data', 'deepprime', f'dp-{data_source}.csv'))
    # drop nan values
    test_data_all = test_data_all.dropna()
    # apply standard scalar
    # cast all numeric columns to float
    test_data_all.iloc[:, 2:26] = test_data_all.iloc[:, 2:26].astype(float)
    # scalar = StandardScaler()
    # test_data_all.iloc[:, 2:26] = scalar.fit_transform(test_data_all.iloc[:, 2:26])

    dp_model = skorch.NeuralNetRegressor(
        DeepPrime(hidden_size, num_layers, num_features, dropout),
        # criterion=nn.MSELoss,
        # optimizer=torch.optim.Adam,
        device=device,
    )

    prediction = dict()
    performance = []

    # Load the models
    for i, model in enumerate(models):
        test_data = test_data_all[test_data_all['fold']==i]
        X_test = test_data.iloc[:, :num_features+2]
        y_test = test_data.iloc[:, -2]
        X_test = preprocess_deep_prime(X_test, source)
        y_test = y_test.values
        y_test = y_test.reshape(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        dp_model.initialize()
        
        dp_model.load_params(f_params=model, f_optimizer=None, f_history=None, f_criterion=None)
        
        y_pred = dp_model.predict(X_test)
        if adjustment == 'log':
            y_pred = np.expm1(y_pred)

        pearson = np.corrcoef(y_test.T, y_pred.T)[0, 1]
        spearman = scipy.stats.spearmanr(y_test, y_pred)[0]

        print(f'Fold {i + 1} Pearson: {pearson}, Spearman: {spearman}')

        prediction[i] = y_pred
        performance.append((pearson, spearman))

    del dp_model    
    torch.cuda.empty_cache()
    
    return prediction

def fine_tune_deepprime(fine_tune_fname: str=None):    
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
        for i in range(5):
            fine_tune = fine_tune_data[fine_tune_data['fold'] != i]
            fold = i + 1
            # load the dp hek293t pe 2 model
            model = DeepPrime(128, 1, 24, 0.05)
            model.load_state_dict(torch.load('models/trained-models/deepprime/dp-hek293t-pe2-fold-1.pt', map_location=device))
            
            # freeze the layers other than head and feature mlps
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
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
                optimizer__lr=0.001,
                max_epochs=500,
                batch_size=1024,
                train_split= skorch.dataset.ValidSplit(cv=5),
                callbacks=[
                    skorch.callbacks.EarlyStopping(patience=30),
                    skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'models/trained-models/deepprime/dp-{data_source}-fold-{fold}.pt', f_optimizer=None, f_history=None, f_criterion=None),
                    skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts , monitor='valid_loss', T_0=10, T_mult=1),
                ]
            )
            
            
            X_fine_tune = fine_tune.iloc[:, :26]
            y_fine_tune = fine_tune.iloc[:, -2]
            
            X_fine_tune = preprocess_deep_prime(X_fine_tune)
            y_fine_tune = y_fine_tune.values
            y_fine_tune = y_fine_tune.reshape(-1, 1)
            y_fine_tune = torch.tensor(y_fine_tune, dtype=torch.float32)
            
            # train the model
            dp_model.fit(X_fine_tune, y_fine_tune)


def deepprime(save_path: str, fine_tune: bool = False) -> skorch.NeuralNet:
    '''
    Returns the DeepPrime model wrapped by skorch
    '''
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = DeepPrime(128, 1, 24, 0.05)
    if fine_tune:
        # load the dp hek293t pe 2 model
        m.load_state_dict(torch.load('models/trained-models/deepprime/dp-hek293t-pe2-fold-1.pt', map_location=device))
        
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

def deepprime_weighted(save_path: str, fine_tune: bool=False) -> skorch.NeuralNet:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = DeepPrime(128, 1, 24, 0.05)
    if fine_tune:
        # load the dp hek293t pe 2 model
        m.load_state_dict(torch.load('models/trained-models/deepprime/dp-hek293t-pe2-fold-1.pt', map_location=device))
        
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