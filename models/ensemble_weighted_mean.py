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
            max_epochs=500,
            module__n_regressors=n_regressors,
            lr=0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=2048,
            train_split=skorch.dataset.ValidSplit(cv=5),
            # early stopping
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=20),
                skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=f'{save_path}-optimizer.pkl', f_history=f'{save_path}-history.json', f_params=f'{save_path}.pkl', event_name='event_cp'),
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
        self.model.load_params(f_params=f'{save_path}-params.pkl')
        
    def initialize(self):
        self.model.initialize()

class EnsembleWeightedMean:
    def __init__(self, optimization: str = True, n_regressors: int = 4, with_features: bool = False):
        """
        Args:
            optimization (str, optional): to use direct optimization or not. Defaults to True.
            n_regressors (int, optional): number of regressors in the ensemble. Defaults to 5.
        """
        self.mlp = [mlp('') for _ in range(5)]
        self.ridge = [ridge_regression() for _ in range(5)]
        self.xgboost = [xgboost() for _ in range(5)]
        self.random_forest = [random_forest() for _ in range(5)]
        self.ensemble = [WeightedMeanSkorch(n_regressors=n_regressors if not with_features else n_regressors + 24) for _ in range(5)]
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
                self.random_forest[i].fit(features, target)
                # save the model
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random-forest-{data_source}-fold-{i+1}.pkl'), 'wb') as f:
                    pickle.dump(self.random_forest[i], f)
            else:
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.random_forest[i] = pickle.load(f)

            if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pkl')):
                print('Training MLP')
                # preprocess the training data
                self.mlp[i] = mlp(save_path=pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}'))
                features = torch.tensor(features, dtype=torch.float32)
                target = torch.tensor(target, dtype=torch.float32)
                self.mlp[i].fit(features, target)
            else:
                self.mlp[i].initialize()
                with open(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'mlp-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                    self.mlp[i].load_params(f_params=f)

            if self.optimization:
                # load or train the ensemble model
                if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}.pkl')):
                    print(f'Training Ensemble, with features: {self.with_features}, optimization: {self.optimization}')
                    self.ensemble[i] = WeightedMeanSkorch(n_regressors=self.n_regressors if not self.with_features else self.n_regressors + 24, save_path=pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}') if not self.with_features else pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features'))
                    # get the predictions from the base models
                    ridge_pred = self.ridge[i].predict(features)
                    xgboost_pred = self.xgboost[i].predict(features)
                    random_forest_pred = self.random_forest[i].predict(features)
                    features = torch.tensor(features, dtype=torch.float32)
                    mlp_pred = self.mlp[i].predict(features)
                    # concatenate the predictions
                    predictions = torch.tensor([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred], dtype=torch.float32).T
                    print(predictions.shape)
                    if self.with_features:
                        predictions = torch.cat((predictions, features), dim=1)
                    # train the ensemble model
                    print(predictions.dtype, target.dtype)
                    self.ensemble[i].fit(predictions, target)
                else:
                    if self.with_features:
                        self.ensemble[i] = WeightedMeanSkorch(n_regressors=self.n_regressors + 24)
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features'))
                    else:
                        self.ensemble[i] = WeightedMeanSkorch(n_regressors=self.n_regressors)
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}'))
                        
            else:
                # weight is the inverse of the error
                ridge_pred = self.ridge[i].predict(features)
                xgboost_pred = self.xgboost[i].predict(features)
                random_forest_pred = self.random_forest[i].predict(data_source, fold=i+1)
                mlp_pred = self.mlp[i].predict(features)
                self.ensemble[i] = [1 / (1 - torch.nn.functional.mse_loss(ridge_pred, target)), 1 / (1 - torch.nn.functional.mse_loss(xgboost_pred, target)), 1 / (1 - torch.nn.functional.mse_loss(random_forest_pred, target)), 1 / (1 - torch.nn.functional.mse_loss(mlp_pred, target))]    

    def test(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]
        performances_pearson = collections.defaultdict(list)
        performances_spearman = collections.defaultdict(list)
        for i in range(5):
            data = dataset[dataset['fold'] == i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            ridge_pred = self.ridge[i].predict(features)
            xgboost_pred = self.xgboost[i].predict(features)
            random_forest_pred = self.random_forest[i].predict(features)
            mlp_pred = self.mlp[i].predict(features)
            
            if self.optimization:
                predictions = torch.tensor([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred], dtype=torch.float32).T
                print(predictions.shape)
                if self.with_features:
                    predictions = torch.cat((predictions, features), dim=1)#
                print(predictions.shape)
                ensemble_predictions = self.ensemble[i].predict(predictions)
            else:
                # weighted mean using the weights from the training
                ensemble_predictions = torch.tensor([ridge_pred, xgboost_pred, random_forest_pred, mlp_pred], dtype=torch.float32).T @ torch.tensor(self.ensemble[i], dtype=torch.float32)

            # record the performance as pearson and spearman correlation
            performances_pearson['ridge'].append(pearsonr(ridge_pred, target)[0])
            performances_pearson['xgboost'].append(pearsonr(xgboost_pred, target)[0])
            performances_pearson['random_forest'].append(pearsonr(random_forest_pred, target)[0])
            performances_pearson['mlp'].append(pearsonr(mlp_pred, target)[0])
            performances_pearson['ensemble'].append(pearsonr(ensemble_predictions, target)[0])

            performances_spearman['ridge'].append(spearmanr(ridge_pred, target)[0])
            performances_spearman['xgboost'].append(spearmanr(xgboost_pred, target)[0])
            performances_spearman['random_forest'].append(spearmanr(random_forest_pred, target)[0])
            performances_spearman['mlp'].append(spearmanr(mlp_pred, target)[0])
            performances_spearman['ensemble'].append(spearmanr(ensemble_predictions, target)[0])

            print(f'Fold {i+1} Pearson: Ridge: {performances_pearson["ridge"][-1]}, XGBoost: {performances_pearson["xgboost"][-1]}, Random Forest: {performances_pearson["random_forest"][-1]}, MLP: {performances_pearson["mlp"][-1]}, Ensemble: {performances_pearson["ensemble"][-1]}')

        return performances_pearson, performances_spearman
