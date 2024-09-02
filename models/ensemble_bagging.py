import numpy as np
from models.conventional_ml_models import mlp, ridge_regression, random_forest, xgboost
import torch
import pickle
from os.path import join as pjoin, isfile
import pandas as pd
import collections
from scipy.stats import pearsonr, spearmanr
from models.deepprime import deepprime, preprocess_deep_prime, WeightedSkorch
from models.pridict import pridict, preprocess_pridict
import skorch
import logging
log = logging.getLogger(__name__)

class EnsembleBagging:
    def __init__(self, n_rounds: int = 3, sample_percentage: float = 0.7):
        """ 
        
        """
        self.n_rounds = n_rounds
        self.base_learners = {
            'xgb': xgboost,
            'mlp': mlp,
            'ridge': ridge_regression,
            'rf': random_forest,
            'dp': deepprime, 
            # 'pd': pridict
        }
        self.dl_models = ['mlp', 'dp', 'pd']
        self.models = []
        self.model_weights = []
        self.sample_percentage = sample_percentage
        # set the random seed
        # self.seed = np.random.seed(42)

        
    def fit(self, data: str, fine_tune: bool = False):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]
        for fold in range(5):
            models = []
            model_weights = []
            data = dataset[dataset['fold'] != fold]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            # each round creates performs the boost on a new set of models
            for i in range(self.n_rounds):
                # create a new set of models
                for ind, base_learner in enumerate(self.base_learners):
                    # sample a subset of the training data with replacement
                    np.random.seed(ind + i * len(self.base_learners))
                    indices = np.random.choice(len(target), int(len(target) * self.sample_percentage), replace=True)
                    data_round = data.iloc[indices]
                    features_round = features[indices]
                    target_round = target[indices]
                    save_path = pjoin('models', 'trained-models', 'ensemble', 'bagging', f'{base_learner}-{data_source}-percentage-{int(self.sample_percentage * 100)}-fold-{fold+1}-round-{i+1}')
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
                            target_round = target_round.view(-1, 1)
                            if base_learner == 'dp':
                                model.fit(preprocess_deep_prime(data_round), target_round)
                            elif base_learner == 'pd':
                                model.fit(preprocess_pridict(data_round), target_round)
                            else:
                                model.fit(features_round, target_round)
                            target_round = target_round.view(-1)
                    else:
                        if isfile(f'{save_path}.pkl'):
                            with open(f'{save_path}.pkl', 'rb') as f:
                                model = pickle.load(f)
                        else:
                            print(f"Training {base_learner}")
                            model.fit(features_round, target_round)
                            with open(f'{save_path}.pkl', 'wb') as f:
                                pickle.dump(model, f)

                    # add the model to the list
                    models.append(model)
                    # use pearson correlation as the weight
                    if base_learner == 'dp':
                        predictions = model.predict(preprocess_deep_prime(data_round)).flatten()
                        # model_weights.append(pearsonr(predictions, target_round)[0])
                    elif base_learner == 'pd':
                        predictions = model.predict(preprocess_pridict(data_round)).flatten()
                    else:
                        predictions = model.predict(features_round).flatten()
                    model_weights.append(abs(spearmanr(predictions, target_round)[0]))

            self.models.append(models)
            # normalize the weights
            model_weights = np.array(model_weights)
            model_weights = model_weights / np.sum(model_weights)
            self.model_weights.append(model_weights)
            

        return self.models, self.model_weights
    
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
            models = self.models[i]
            model_weights = self.model_weights[i]

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
                
                predictions = model.predict(features).flatten()
                performances_pearson[base_learner].append(pearsonr(predictions, target)[0])
                performances_spearman[base_learner].append(spearmanr(predictions, target)[0])

            # aggregated predictions
            agg_predictions = np.zeros_like(target)
            for model, weight in zip(models, model_weights):
                if isinstance(model, WeightedSkorch):
                    predictions = model.predict(preprocess_deep_prime(data)).flatten()
                else:
                    predictions = model.predict(features).flatten()
                agg_predictions += weight * predictions

            # calculate the performance
            performances_pearson['bag'].append(pearsonr(agg_predictions, target)[0])
            performances_spearman['bag'].append(spearmanr(agg_predictions, target)[0])
        
        return performances_pearson, performances_spearman
    
    
def predict(data: str):
    """
    Perform the prediction using the trained models
    Produce predictions for the full dataset using corresponding fold models
    """
    model = EnsembleBagging()
    data_source = '-'.join(data.split('-')[1:]).split('.')[0]
    model.fit(f'ensemble-{data_source}.csv')
    dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', f'ensemble-{data_source}.csv'))
    predictions = dict()
    for i in range(5):
        alphas = model.model_weights[i]
        models = model.models[i]

        data = dataset[dataset['fold'] == i]
        features = data.iloc[:, 2:26].values
        target = data.iloc[:, -2].values
        features = torch.tensor(features, dtype=torch.float32)

        # aggregated predictions
        agg_predictions = np.zeros(len(target))

        for m, alpha in zip(models, alphas):
            if isinstance(m, WeightedSkorch) or isinstance(m, skorch.NeuralNet):
                prediction = m.predict(preprocess_deep_prime(data)).flatten()
            else:
                prediction = m.predict(features).flatten()
            agg_predictions += alpha * prediction

        predictions[i] = agg_predictions

    return predictions

def predict_df(data: pd.DataFrame, cell_line: str, pe: str) -> np.ndarray:
    """
    Perform the prediction using the trained models on a DataFrame

    Args:
        data (pd.DataFrame): the data to perform the prediction on
        cell_line (str): the cell line to predict
        pe (str): the pe to predict

    Returns:
        np.ndarray: the predicted editing efficiency
    """
    model = EnsembleBagging()
    data_sources = ['pd', 'dp', 'dp_small']
    predictions = []

    log.log(msg=f'Predicting on {len(data)} sequences', level=logging.INFO)

    for data_source in data_sources:
        if isfile(pjoin('models', 'data', 'ensemble', f'ensemble-{data_source}-{cell_line}-{pe}.csv')):
            model.fit(f'ensemble-{data_source}-{cell_line}-{pe}.csv')
        else:
            log.error(f'No data found for {data_source}-{cell_line}-{pe}')
            continue

        log.log(msg=f'Running ensemble prediction on {len(data)}', level=logging.INFO)
        for i in range(5):
            alphas = model.model_weights[i]
            models = model.models[i]

            features = data.iloc[:, 2:26].values
            features = features.astype(np.float32)

            # aggregated predictions
            agg_predictions = np.zeros(len(
                data.iloc[:, -2].values
            ))

            for m, alpha in zip(models, alphas):
                if isinstance(m, WeightedSkorch) or isinstance(m, skorch.NeuralNet):
                    prediction = m.predict(preprocess_deep_prime(data)).flatten()
                else:
                    prediction = m.predict(features).flatten()
                agg_predictions += alpha * prediction

            predictions.append(agg_predictions)

    # calculate the mean of the predictions
    predictions = np.mean(predictions, axis=0)

    return predictions