import numpy as np
from models.conventional_ml_models import mlp_weighted, ridge_regression, random_forest, xgboost
from models.deepprime import deepprime_weighted, preprocess_deep_prime, WeightedSkorch
from models.pridict import pridict_weighted, preprocess_pridict
import torch
import pickle
from os.path import join as pjoin, isfile
import pandas as pd
import collections
from scipy.stats import pearsonr, spearmanr
import sklearn
import collections

class EnsembleAdaBoost:
    def __init__(self, n_rounds: int = 1, threshold=0.5, power:int = 3):
        """ 
        
        """
        self.n_rounds = n_rounds
        self.base_learners = {
            'xgb': xgboost,
            'mlp': mlp_weighted,
            'ridge': ridge_regression,
            'rf': random_forest,
            'dp': deepprime_weighted,
            'pd': pridict_weighted
        }
        self.dl_models = ['mlp', 'dp', 'pd']
        self.models = []
        self.alphas = []
        self.threshold = threshold
        self.power = power
        # set the random seed
        np.random.seed(42)
        torch.manual_seed(42)
        
    def fit(self, data: str, fine_tune: bool = False):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        # dataset = dataset.sample(frac=percentage, random_state=42)
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]
        for fold in range(5):
            data = dataset[dataset['fold'] != fold]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
            target_np = np.array(target).flatten()
            # deal with the case where the target is 0
            target_np[target_np == 0] += 1e-6
            sample_weights = np.ones(len(target))

            # # aggravated predictions
            # agg_predictions = np.zeros(len(target))

            # each round creates performs the boost on a new set of models
            for i in range(self.n_rounds):
                models = []
                alphas = []
                # create a new set of models
                for base_learner in self.base_learners:
                    save_path = pjoin('models', 'trained-models', 'ensemble', 'adaboost', f'{base_learner}-{data_source}-fold-{fold+1}-round-{i+1}-threshold-{self.threshold}-power-{self.power}')
                    print(f"Round {i+1} {base_learner}")
                    if base_learner in self.dl_models:
                        model = self.base_learners[base_learner](save_path=save_path, fine_tune=fine_tune)
                    else:
                        model = self.base_learners[base_learner]()
                    # train or load the model
                    if base_learner in self.dl_models:
                        if isfile(f'{save_path}.pt'):
                            model.initialize()
                            model.load_params(f_params=f'{save_path}.pt')
                        else:
                            print(f"Training {base_learner}")
                            target = target.view(-1, 1)
                            # sample weights need to be applied to all the features
                            sample_weights = torch.tensor(sample_weights, dtype=torch.float32).view(-1, 1)
                            if base_learner == 'dp':
                                model.fit(preprocess_deep_prime(data, sample_weight=sample_weights), target)
                            elif base_learner == 'pd':
                                model.fit(preprocess_pridict(data, sample_weight=sample_weights), target)
                            else:
                                feature_X = {
                                    'x': features,
                                    'sample_weight': sample_weights
                                }
                                model.fit(feature_X, target)
                            sample_weights = sample_weights.view(-1)
                            sample_weights = sample_weights.numpy().flatten()
                            target = target.view(-1)
                    else:
                        if isfile(f'{save_path}.pkl'):
                            with open(f'{save_path}.pkl', 'rb') as f:
                                model = pickle.load(f)
                        else:
                            print(f"Training {base_learner}")
                            model.fit(features, target, sample_weight=sample_weights)
                            with open(f'{save_path}.pkl', 'wb') as f:
                                pickle.dump(model, f)

                    # make predictions
                    if base_learner == 'dp':
                        predictions = model.predict(preprocess_deep_prime(data)).flatten()
                    elif base_learner == 'pd':
                        predictions = model.predict(preprocess_pridict(data)).flatten()
                    else:
                        predictions = model.predict(features).flatten()
                    predictions = np.array(predictions)
                    # calculate the correlation between the predictions and the target as the model weight
                    alpha = abs(spearmanr(predictions, target_np)[0])
                    # calculate relative error for each sample
                    error = np.abs(predictions - target_np)
                    error = error / target_np
                    # calculate the error rate
                    error_rate = np.sum(error > self.threshold) / len(error)
                    beta = np.power(error_rate, self.power)
                    # update the sample weights by multiplying the error rate for correct predictions
                    error = [beta if e <= self.threshold else 1 for e in error]
                    sample_weights = sample_weights * error
                    # normalize the weights to have a mean of 1
                    sample_weights = sample_weights / np.mean(sample_weights)
                    # make sure no weight is less than 1e-6
                    sample_weights = np.maximum(sample_weights, 1e-4)
                    print(sample_weights)
                    # add the model to the list
                    models.append(model)
                    alphas.append(alpha)

            # normalize the weights
            alphas = np.array(alphas)
            alphas = alphas / np.sum(alphas)
            self.models.append(models)
            self.alphas.append(alphas)


        return self.models, self.alphas
    
    def tune(self, data: str):
        self.base_learners = {
            'xgb': xgboost,
            'mlp': mlp_weighted,
            'ridge': ridge_regression,
            'rf': random_forest,
        }
        # tune the hyperparameters power and threshold
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]

        # only use the first fold for tuning
        fold = 0
        data = dataset[dataset['fold'] != fold]
        features = data.iloc[:, 2:26].values
        target = data.iloc[:, -2].values
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        target_np = np.array(target).flatten()

        data_test = dataset[dataset['fold'] == fold]
        features_test = data_test.iloc[:, 2:26].values
        target_test = data_test.iloc[:, -2].values
        features_test = torch.tensor(features_test, dtype=torch.float32)
        target_test = torch.tensor(target_test, dtype=torch.float32)
        target_np_test = np.array(target_test).flatten()

        # deal with the case where the target is 0
        target_np[target_np == 0] += 1e-6
        
        self.n_rounds = 10
        configurations = {
            'power': [1, 2, 3],
            'threshold': [0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
            'rounds': [1, 3, 5, 10]
        }
        param_grid = sklearn.model_selection.ParameterGrid(configurations)
        output = []

        for param in param_grid:
            self.power = param['power']
            self.threshold = param['threshold']
            self.n_rounds = param['rounds']
            param['pearson'] = []
            param['spearman'] = []

            # each round creates performs the boost on a new set of models
            for run in range(3):
                sample_weights = np.ones(len(target))
                for i in range(self.n_rounds):
                    models = []
                    alphas = []
                    # create a new set of models
                    for base_learner in self.base_learners:
                        save_path = pjoin('models', 'trained-models', 'ensemble', 'adaboost', f'{base_learner}-{data_source}-fold-{fold+1}-round-{i+1}-threshold-{self.threshold}-power-{self.power}')
                        if base_learner in self.dl_models:
                            model = self.base_learners[base_learner](save_path=save_path)
                        else:
                            model = self.base_learners[base_learner]()
                        # train or load the model
                        if base_learner in self.dl_models:
                            if isfile(f'{save_path}.pt'):
                                model.initialize()
                                model.load_params(f_params=f'{save_path}.pt')
                            else:
                                print(f"Training {base_learner}")
                                target = target.view(-1, 1)
                                # sample weights need to be applied to all the features
                                sample_weights = torch.tensor(sample_weights, dtype=torch.float32).view(-1, 1)
                                feature_X = {
                                    'x': features,
                                    'sample_weight': sample_weights
                                }
                                sample_weights = sample_weights.view(-1)
                                sample_weights = sample_weights.numpy().flatten()
                                model.fit(feature_X, target)
                                target = target.view(-1)
                        else:
                            if isfile(f'{save_path}.pkl'):
                                with open(f'{save_path}.pkl', 'rb') as f:
                                    model = pickle.load(f)
                            else:
                                print(f"Training {base_learner}")
                                model.fit(features, target, sample_weight=sample_weights)
                                with open(f'{save_path}.pkl', 'wb') as f:
                                    pickle.dump(model, f)

                        # make predictions
                        predictions = model.predict(features).flatten()
                        predictions = np.array(predictions)
                        # calculate the correlation between the predictions and the target as the model weight
                        alpha = spearmanr(predictions, target_np)[0]
                        # calculate relative error for each sample
                        error = np.abs(predictions - target_np)
                        error = error / target_np
                        # calculate the error rate
                        error_rate = np.sum(error > self.threshold) / len(error)
                        beta = np.power(error_rate, self.power)
                        # update the sample weights by multiplying the error rate for correct predictions
                        error = [beta if e <= self.threshold else 1 for e in error]
                        sample_weights = sample_weights * error
                        # normalize the weights to have a mean of 1
                        sample_weights = sample_weights / np.mean(sample_weights)
                        # make sure no weight is less than 1e-6
                        sample_weights = np.maximum(sample_weights, 1e-6)
                        # add the model to the list
                        models.append(model)
                        alphas.append(alpha)

                # normalize the weights
                alphas = np.array(alphas)
                alphas = alphas / np.sum(alphas)
                # perform the prediction on the test set
                agg_predictions = np.zeros(len(target_test))

                for model, alpha in zip(models, alphas):
                    predictions = model.predict(features_test).flatten()
                    agg_predictions += alpha * predictions

                # calculate the performance
                performances_pearson = pearsonr(agg_predictions, target_np_test)[0]
                performances_spearman = spearmanr(agg_predictions, target_np_test)[0]
                if run == 0:
                    print(f"Power: {self.power}, Threshold: {self.threshold}, Rounds: {self.n_rounds}, Pearson: {performances_pearson}, Spearman: {performances_spearman}")

                param['pearson'].append(performances_pearson)
                param['spearman'].append(performances_spearman)

            output.append(param)
            
        self.base_learners = {
            'xgb': xgboost,
            'mlp': mlp_weighted,
            'ridge': ridge_regression,
            'rf': random_forest,
            'dp': deepprime
        }

        return output

    
    def test(self, data: str):
        """Perform the prediction using the trained models
        
        Args:
            data (str): the data to perform the prediction on
        """
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        performances_pearson = collections.defaultdict(list)
        performances_spearman = collections.defaultdict(list)
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]

        for i in range(5):
            alphas = self.alphas[i].flatten()
            models = self.models[i]

            data = dataset[dataset['fold'] == i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            # run the performance using base learners for comparison
            for base_learner in self.base_learners:
                save_path = pjoin('models', 'trained-models', 'ensemble', 'weighted-mean', f'{base_learner}-{data_source}-fold-{i+1}')
                if base_learner in self.dl_models:
                    model = self.base_learners[base_learner](save_path=save_path)
                    model.initialize()
                    model.load_params(f_params=f'{save_path}.pt')
                else:
                    with open(f'{save_path}.pkl', 'rb') as f:
                        model = pickle.load(f)
                if base_learner == 'dp':
                    predictions = model.predict(preprocess_deep_prime(data)).flatten()
                else:
                    predictions = model.predict(features).flatten()
                performances_pearson[base_learner].append(pearsonr(predictions, target)[0])
                performances_spearman[base_learner].append(spearmanr(predictions, target)[0])

            # aggregated predictions
            agg_predictions = np.zeros(len(target))

            for model, alpha in zip(models, alphas):
                if isinstance(model, WeightedSkorch):
                    predictions = model.predict(preprocess_deep_prime(data)).flatten()
                else:
                    predictions = model.predict(features).flatten()
                agg_predictions += alpha * predictions

            # calculate the performance
            performances_pearson['ada'].append(pearsonr(agg_predictions, target)[0])
            performances_spearman['ada'].append(spearmanr(agg_predictions, target)[0])
        
        return performances_pearson, performances_spearman
    
def predict(data: str):
    """
    Perform the prediction using the trained models
    Produce predictions for the full dataset using corresponding fold models
    """
    model = EnsembleAdaBoost()
    data_source = '-'.join(data.split('-')[1:]).split('.')[0]
    model.fit(f'ensemble-{data_source}.csv')
    dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', f'ensemble-{data_source}.csv'))
    # drop nan
    dataset = dataset.dropna()
    predictions = dict()
    for i in range(5):
        alphas = model.alphas[i].flatten()
        models = model.models[i]

        data = dataset[dataset['fold'] == i]
        features = data.iloc[:, 2:26].values
        target = data.iloc[:, -2].values
        features = torch.tensor(features, dtype=torch.float32)

        # aggregated predictions
        agg_predictions = np.zeros(len(target))

        for m, alpha in zip(models, alphas):
            if isinstance(m, WeightedSkorch):
                prediction = m.predict(preprocess_deep_prime(data)).flatten()
            else:
                prediction = m.predict(features).flatten()
            agg_predictions += alpha * prediction

        predictions[i] = agg_predictions

    return predictions