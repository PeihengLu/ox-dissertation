"""
Weighted mean implemented using a single layer neural network.
"""
import torch
import skorch
import collections
from scipy.stats import pearsonr, spearmanr

from typing import List
from models.conventional_ml_models import mlp, ridge_regression, lasso_regression, xgboost, random_forest
import pickle
import pandas as pd
import numpy as np

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
            save_path (str, optional): path to save the model. Defaults to None.
        """
        self.model = skorch.NeuralNetRegressor(
            module=WeightedMeanModel,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            max_epochs=100,
            module__n_regressors=n_regressors,
            lr=0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=2048,
            train_split=skorch.dataset.ValidSplit(cv=5),
            # early stopping
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=20),
                skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=None, f_history=None, f_params=f'{save_path}.pt', event_name='event_cp'),
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
        self.model.load_params(f_params=f'{save_path}')
        
    def initialize(self):
        self.model.initialize()

class EnsembleWeightedMean:
    def __init__(self, optimization: str = True, n_regressors: int = 4, with_features: bool = False):
        """
        Args:
            optimization (str, optional): to use direct optimization or not. Defaults to True.
            n_regressors (int, optional): number of regressors in the ensemble. Defaults to 5.
        """
        self.mlp = [None for _ in range(5)]
        self.ridge = [ridge_regression() for _ in range(5)]
        self.xgboost = [xgboost() for _ in range(5)]
        self.random_forest = [random_forest() for _ in range(5)]
        self.ensemble = [None for _ in range(5)]
        self.optimization = optimization
        self.n_regressors = n_regressors
        self.with_features = with_features

    # fit would load the models if trained, if not, it would train the models
    def fit(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        # dataset = dataset.sample(frac=percentage, random_state=42)
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]

        for i in range(5):
            data = dataset[dataset['fold'] != i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            
            self.ensemble[i] = WeightedMeanSkorch(n_regressors=self.n_regressors if not self.with_features else self.n_regressors + 24, save_path=pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}') if not self.with_features else pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features'))
            # reshape target
            target = target.reshape(-1, 1)
            # load or train the base models
            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ridge-{data_source}-fold-{i+1}.pkl')):
                print('Training Ridge')
                self.ridge[i].fit(features, target)
                # save the model
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ridge-{data_source}-fold-{i+1}.pkl'), 'wb') as f:
                    pickle.dump(self.ridge[i], f)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ridge-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.ridge[i] = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'xgboost-{data_source}-fold-{i+1}.pkl')):
                print('Training XGBoost')
                self.xgboost[i].fit(features, target)
                # save the model
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'wb') as f:
                    pickle.dump(self.xgboost[i], f)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.xgboost[i] = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random_forest-{data_source}-fold-{i+1}.pkl')):
                print('Training Random Forest')
                # reshape target
                target = target.reshape(-1)
                self.random_forest[i].fit(features, target)
                # reverse the reshape
                target = target.reshape(-1, 1)
                # save the model
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'wb') as f:
                    pickle.dump(self.random_forest[i], f)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.random_forest[i] = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pt')):
                print('Training MLP')
                # preprocess the training data
                self.mlp[i] = mlp(save_path=pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}'))
                self.mlp[i].fit(features, target)
            else:
                self.mlp[i] = mlp('')
                self.mlp[i].initialize()
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pt'), 'rb') as f:
                    self.mlp[i].load_params(f_params=f)
                
            ridge_pred = self.ridge[i].predict(features).flatten()
            xgboost_pred = self.xgboost[i].predict(features).flatten()
            random_forest_pred = self.random_forest[i].predict(features).flatten()
            mlp_pred = self.mlp[i].predict(features).flatten()
            predictions = torch.tensor(np.array([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred]), dtype=torch.float32).T

            if self.optimization:
                if self.with_features:
                    predictions = np.concatenate((predictions, features), axis=1)
                    if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features.pt')):
                        print('Training Ensemble')
                        self.ensemble[i].fit(predictions, target)
                    else:
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features.pt'))
                else:
                    if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}.pt')):
                        print('Training Ensemble')
                        self.ensemble[i].fit(predictions, target)
                    else:
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}.pt'))
            else:
                target = target.flatten()
                self.ensemble[i] = [pearsonr(ridge_pred, target)[0], pearsonr(xgboost_pred, target)[0], pearsonr(random_forest_pred, target)[0], pearsonr(mlp_pred, target)[0]] 
                # normalize the weights
                self.ensemble[i] = self.ensemble[i] / np.sum(self.ensemble[i])

    def test(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]
        performances_pearson = collections.defaultdict(list)
        performances_spearman = collections.defaultdict(list)
        performances_pearson_train = collections.defaultdict(list)
        performances_spearman_train = collections.defaultdict(list)
        for i in range(5):
            data = dataset[dataset['fold'] == i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            data_train = dataset[dataset['fold'] != i]
            features_train = data_train.iloc[:, 2:26].values
            target_train = data_train.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            features_train = torch.tensor(features_train, dtype=torch.float32)
            target_train = torch.tensor(target_train, dtype=torch.float32)
            
            ridge_pred = self.ridge[i].predict(features).flatten()
            xgboost_pred = self.xgboost[i].predict(features).flatten()
            random_forest_pred = self.random_forest[i].predict(features).flatten()
            mlp_pred = self.mlp[i].predict(features).flatten()
            
            ridge_pred_train = self.ridge[i].predict(features_train).flatten()
            xgboost_pred_train = self.xgboost[i].predict(features_train).flatten()
            random_forest_pred_train = self.random_forest[i].predict(features_train).flatten()
            mlp_pred_train = self.mlp[i].predict(features_train).flatten()
            
            if self.optimization:
                predictions = torch.tensor(np.array([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred]), dtype=torch.float32).T
                predictions_train = torch.tensor(np.array([ridge_pred_train, xgboost_pred_train, random_forest_pred_train, mlp_pred_train]), dtype=torch.float32).T
                if self.with_features:
                    predictions = torch.cat((predictions, features), dim=1)
                    predictions_train = torch.cat((predictions_train, features_train), dim=1)
                ensemble_predictions = self.ensemble[i].predict(predictions).flatten()
                ensemble_predictions_train = self.ensemble[i].predict(predictions_train).flatten()
            else:
                # weighted mean using the weights from the training
                ensemble_predictions = torch.tensor(np.array([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred]), dtype=torch.float32).T @ torch.tensor(self.ensemble[i], dtype=torch.float32)
                ensemble_predictions_train = torch.tensor(np.array([ridge_pred_train, xgboost_pred_train, random_forest_pred_train, mlp_pred_train]), dtype=torch.float32).T @ torch.tensor(self.ensemble[i], dtype=torch.float32)
                ensemble_predictions = ensemble_predictions.flatten()
                ensemble_predictions_train = ensemble_predictions_train.flatten()

            # record the performance as pearson and spearman correlation
            performances_pearson['ridge'].append(pearsonr(ridge_pred, target)[0])
            performances_pearson['xgb'].append(pearsonr(xgboost_pred, target)[0])
            performances_pearson['rf'].append(pearsonr(random_forest_pred, target)[0])
            performances_pearson['mlp'].append(pearsonr(mlp_pred, target)[0])
            performances_pearson['ensemble'].append(pearsonr(ensemble_predictions, target)[0])

            performances_spearman['ridge'].append(spearmanr(ridge_pred, target)[0])
            performances_spearman['xgb'].append(spearmanr(xgboost_pred, target)[0])
            performances_spearman['rf'].append(spearmanr(random_forest_pred, target)[0])
            performances_spearman['mlp'].append(spearmanr(mlp_pred, target)[0])
            performances_spearman['ensemble'].append(spearmanr(ensemble_predictions, target)[0])
            
            performances_pearson_train['ridge'].append(pearsonr(ridge_pred_train, target_train)[0])
            performances_pearson_train['xgb'].append(pearsonr(xgboost_pred_train, target_train)[0])
            performances_pearson_train['rf'].append(pearsonr(random_forest_pred_train, target_train)[0])
            performances_pearson_train['mlp'].append(pearsonr(mlp_pred_train, target_train)[0])
            performances_pearson_train['ensemble'].append(pearsonr(ensemble_predictions_train, target_train)[0])
            
            performances_spearman_train['ridge'].append(spearmanr(ridge_pred_train, target_train)[0])
            performances_spearman_train['xgb'].append(spearmanr(xgboost_pred_train, target_train)[0])
            performances_spearman_train['rf'].append(spearmanr(random_forest_pred_train, target_train)[0])
            performances_spearman_train['mlp'].append(spearmanr(mlp_pred_train, target_train)[0])
            performances_spearman_train['ensemble'].append(spearmanr(ensemble_predictions_train, target_train)[0])
            
            # rename ensemble according to the optimization and data source
            if self.optimization:
                if self.with_features:
                    ensemble_name = 'opt-f'
                else:
                    ensemble_name = 'opt'
            else:
                ensemble_name = 'pwm'
            # rename the directory name for ensemble
            for performance in [performances_pearson, performances_spearman, performances_pearson_train, performances_spearman_train]:
                performance[ensemble_name] = performance.pop('ensemble')

        return performances_pearson, performances_spearman, performances_pearson_train, performances_spearman_train
