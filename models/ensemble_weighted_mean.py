"""
Weighted mean implemented using a single layer neural network.
"""
import torch
import skorch

from typing import List
from conventional_ml_models import mlp, ridge_regression, lasso_regression, xgboost, random_forest
import pickle
import pandas as pd

from os.path import isfile, join as pjoin

class WeightedMeanModel(torch.nn.Module):
    def __init__(self, n_regressors: int):
        """initialize the model

        Args:
            n_regressors (int): number of regressors in the ensemble
        """
        super(WeightedMeanModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=n_regressors, out_features=1, bias=False)

    def forward(self, X):
        return self.linear(X)
    
class WeightedMeanSkorch():
    def __init__(self, n_regressors: int, save_path: str = None):
        """initialize the model

        Args:
            n_regressors (int): number of regressors in the ensemble
        """
        self.model = skorch.NeuralNetRegressor(
            module=WeightedMeanModel,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            max_epochs=500,
            module__n_regressors=n_regressors,
            lr=0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=2048,
            train_split=skorch.dataset.ValidSplit(cv=5),
            # early stopping
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=50),
                skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=f'{save_path}-optimizer.pkl', f_history=f'{save_path}-history.json', f_params=f'{save_path}-params.pkl', event_name='event_cp'),
                # skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=20, T_mult=1),
            ]
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_params(f_params=path)

    def load(self, save_path: str):
        self.model.load_params(f_params=f'{save_path}-params.pkl', f_optimizer=f'{save_path}-optimizer.pkl', f_history=f'{save_path}-history.json')
        
    def initialize(self):
        self.model.initialize()

class EnsembleWeightedMean:
    def __init__(self):
        self.mlp = mlp()
        self.ridge = ridge_regression()
        self.lasso = lasso_regression()
        self.xgboost = xgboost()
        self.random_forest = random_forest()
        self.models = [self.mlp, self.ridge, self.lasso, self.xgboost, self.random_forest]
        self.ensemble = WeightedMeanSkorch(n_regressors=5)

    # fit would load the models if trained, if not, it would train the models
    def fit(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        # dataset = dataset.sample(frac=percentage, random_state=42)
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]

        for i in range(5):
            features = dataset.iloc[:, 2:26].values
            target = dataset.iloc[:, -2].values
            # load or train the base models
            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'lasso-{data_source}-fold-{i+1}.pkl')):
                print('Training Lasso')
                self.lasso.fit(data_source, fold=i+1)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'lasso-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.lasso = pickle.load(f)
            
            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ridge-{data_source}-fold-{i+1}.pkl')):
                print('Training Ridge')
                self.ridge.fit(data_source, fold=i+1)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ridge-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.ridge = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'xgboost-{data_source}-fold-{i+1}.pkl')):
                print('Training XGBoost')
                self.xgboost.fit(data_source, fold=i+1)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.xgboost = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random-forest-{data_source}-fold-{i+1}.pkl')):
                print('Training Random Forest')
                self.random_forest.fit(data_source, fold=i+1)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random-forest-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.random_forest = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pkl')):
                print('Training MLP')
                # preprocess the training data
                
                self.mlp.fit(data_source, fold=i+1)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.mlp = pickle.load(f)