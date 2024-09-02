import torch
import skorch
import collections
from scipy.stats import pearsonr, spearmanr

from typing import List
from models.conventional_ml_models import mlp, ridge_regression, lasso_regression, xgboost, random_forest
from models.deepprime import deepprime, preprocess_deep_prime, WeightedSkorch
from models.pridict import pridict, preprocess_pridict
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
    def __init__(self, optimization: str = True, n_regressors: int = 5, with_features: bool = False):
        """
        Args:
            optimization (str, optional): to use direct optimization or not. Defaults to True.
            n_regressors (int, optional): number of regressors in the ensemble. Defaults to 5.
        """
        self.optimization = optimization
        self.with_features = with_features
        self.models = []
        self.base_learners = {
            'ridge': ridge_regression,
            'xgb': xgboost,
            'rf': random_forest,
            'mlp': mlp,
            'dp': deepprime,
            'pd': pridict
        }
        self.n_regressors = len(self.base_learners)
        self.ensemble = [None for _ in range(self.n_regressors)]
        self.dl_models = ['mlp', 'dp', 'pd']

    # fit would load the models if trained, if not, it would train the models
    def fit(self, data: str, fine_tune: bool=False):
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

            predictions = []
            models_fold = []
            
            for base_learner in self.base_learners:
                save_path = pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'{base_learner}-{data_source}-fold-{i+1}')
                if base_learner in self.dl_models:
                    model = self.base_learners[base_learner](save_path=save_path, fine_tune=fine_tune)
                    if isfile(f'{save_path}.pt'):
                        model.initialize()
                        model.load_params(f_params=f'{save_path}.pt')
                    else:
                        print(f"Training {base_learner}")
                        # reshape the target
                        target = target.view(-1, 1)
                        if base_learner == 'dp':
                            model.fit(preprocess_deep_prime(data), target)
                        elif base_learner == 'pd':
                            model.fit(preprocess_pridict(data), target)
                        else:
                            model.fit(features, target)
                        target = target.view(-1)
                        model.save_params(f_params=f'{save_path}.pt')
                else:
                    model = self.base_learners[base_learner]()
                    if isfile(f'{save_path}.pkl'):
                        with open(f'{save_path}.pkl', 'rb') as f:
                            model = pickle.load(f)
                    else:
                        print(f"Training {base_learner}")
                        model.fit(features, target)
                        with open(f'{save_path}.pkl', 'wb') as f:
                            pickle.dump(model, f)

                if base_learner == 'dp':
                    predictions.append(model.predict(preprocess_deep_prime(data)).flatten())
                elif base_learner == 'pd':
                    predictions.append(model.predict(preprocess_pridict(data)).flatten())    
                else:
                    predictions.append(model.predict(features).flatten())
                models_fold.append(model)

            self.models.append(models_fold)
            predictions = np.array(predictions).T
            
            if self.optimization:
                target = target.reshape(-1, 1)
                if self.with_features:
                    predictions = np.concatenate((predictions, features), axis=1)
                    predictions = torch.tensor(predictions, dtype=torch.float32)
                    if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features.pt')):
                        print('Training Ensemble')
                        self.ensemble[i].fit(predictions, target)
                    else:
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}-features.pt'))
                else:
                    predictions = torch.tensor(predictions, dtype=torch.float32)
                    if not isfile(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}.pt')):
                        print('Training Ensemble')
                        self.ensemble[i].fit(predictions, target)
                    else:
                        self.ensemble[i].initialize()
                        self.ensemble[i].load(pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'ensemble-{data_source}-fold-{i+1}.pt'))
            else:
                target = target.flatten()
                self.ensemble[i] = []
                # using the pearson correlation as the weight
                for j in range(self.n_regressors):
                    self.ensemble[i].append(abs(spearmanr(predictions[:, j], target)[0]))
                    print(f"Weight for model {j}: {self.ensemble[i][-1]}")
                self.ensemble[i] = np.array(self.ensemble[i])
                # normalize the weights
                self.ensemble[i] = self.ensemble[i] / np.sum(self.ensemble[i])

    def test(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        performances_pearson = collections.defaultdict(list)
        performances_spearman = collections.defaultdict(list)

        for i in range(5):
            data = dataset[dataset['fold'] == i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            # data_train = dataset[dataset['fold'] != i]
            # features_train = data_train.iloc[:, 2:26].values
            # target_train = data_train.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            # features_train = torch.tensor(features_train, dtype=torch.float32)
            # target_train = torch.tensor(target_train, dtype=torch.float32)
            
            predictions = []
            for base_learner in self.models[i]:
                if base_learner == 'dp':
                    predictions.append(base_learner.predict(preprocess_deep_prime(data)).flatten())
                elif base_learner == 'pd':
                    predictions.append(base_learner.predict(preprocess_pridict(data)).flatten())
                else:
                    predictions.append(base_learner.predict(features).flatten())
            
            if self.optimization:
                predictions = torch.tensor(predictions, dtype=torch.float32).T
                if self.with_features:
                    predictions = torch.cat((predictions, features), dim=1)
                ensemble_predictions = self.ensemble[i].predict(predictions).flatten()
            else:
                # weighted mean using the weights from the training
                ensemble_predictions = torch.tensor(predictions, dtype=torch.float32).T @ torch.tensor(self.ensemble[i], dtype=torch.float32)
                ensemble_predictions = ensemble_predictions.flatten()

            if self.optimization:
                predictions = np.array(predictions).T
            # record the performance as pearson and spearman correlation
            for ind, base_learner in enumerate(self.base_learners):
                base_learner_name = base_learner
                performances_pearson[base_learner_name].append(pearsonr(predictions[ind], target)[0])
                performances_spearman[base_learner_name].append(spearmanr(predictions[ind], target)[0])
            performances_pearson['ensemble'].append(pearsonr(ensemble_predictions, target)[0])

            for ind, base_learner in enumerate(self.base_learners):
                base_learner_name = base_learner
                performances_pearson[base_learner_name].append(pearsonr(predictions[ind], target)[0])
                performances_spearman[base_learner_name].append(spearmanr(predictions[ind], target)[0])
            performances_spearman['ensemble'].append(spearmanr(ensemble_predictions, target)[0])
            
            # rename ensemble according to the optimization and data source
            if self.optimization:
                if self.with_features:
                    ensemble_name = 'opt-f'
                else:
                    ensemble_name = 'opt'
            else:
                ensemble_name = 'pwm'
            # rename the directory name for ensemble
            for performance in [performances_pearson, performances_spearman]:
                performance[ensemble_name] = performance.pop('ensemble')

        return performances_pearson, performances_spearman
    

def predict(data: str):
    """
    Perform the prediction using the trained models
    Produce predictions for the full dataset using corresponding fold models
    """
    model = EnsembleWeightedMean(optimization=False)
    data_source = '-'.join(data.split('-')[1:]).split('.')[0]
    model.fit(f'ensemble-{data_source}.csv')
    dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', f'ensemble-{data_source}.csv'))
    dataset = dataset.dropna()
    predictions = dict()
    for i in range(5):
        models = model.models[i]

        data = dataset[dataset['fold'] == i]
        features = data.iloc[:, 2:26].values
        target = data.iloc[:, -2].values
        features = torch.tensor(features, dtype=torch.float32)

        # aggregated predictions
        prediction_list = []

        for m in models:
            if isinstance(m, WeightedSkorch) or isinstance(m, skorch.NeuralNet):
                prediction = m.predict(preprocess_deep_prime(data)).flatten()
            else:
                prediction = m.predict(features).flatten()
            prediction_list.append(prediction)

        ensemble_predictions = torch.tensor(prediction_list, dtype=torch.float32).T @ torch.tensor(model.ensemble[i], dtype=torch.float32)
        ensemble_predictions = ensemble_predictions.flatten()
        predictions[i] = ensemble_predictions
        

    return predictions    
