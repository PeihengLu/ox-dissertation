import numpy as np
from models.conventional_ml_models import mlp_weighted, ridge_regression, random_forest, xgboost
import torch
import pickle
from os.path import join as pjoin, isfile
import pandas as pd

class EmsenbleAdaBoost:
    def __init__(self, n_rounds: int = 10, learning_rate=1.0):
        """ 
        
        """
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.base_learners = {
            'mlp_weighted': mlp_weighted,
            'ridge_regression': ridge_regression,
            'random_forest': random_forest,
            'xgboost': xgboost
        }
        self.dl_models = ['mlp_weighted']
        self.models = []
        self.alphas = []
        self.sample_weights = []
        # set the random seed
        np.random.seed(42)
        torch.manual_seed(42)
        
    def fit(self, data: str):
        dataset = pd.read_csv(pjoin('models', 'data', 'ensemble', data))
        # dataset = dataset.sample(frac=percentage, random_state=42)
        cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
        data_source = '-'.join(data.split('-')[1:]).split('.')[0]
        for fold in range(5):
            self.sample_weights = np.ones(len(dataset)) / len(dataset)
            data = dataset[dataset['fold'] != i]
            features = data.iloc[:, 2:26].values
            target = data.iloc[:, -2].values
            features = torch.tensor(features, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

            # aggravated predictions
            agg_predictions = np.zeros(len(target))

            # each round creates performs the boost on a new set of models
            for i in range(self.n_rounds):
                # create a new set of models
                for base_learner in self.base_learners:
                    save_path = pjoin('models', 'trained-models', 'ensemble', 'adaboost', f'{base_learner}-{data_source}-fold-{fold}-round-{i}')
                    if base_learner == 'mlp_weighted':
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
                            model.fit(features, target, sample_weight=self.sample_weights)
                            target = target.view(-1)
                    else:
                        if isfile(f'{save_path}.pkl'):
                            with open(f'{save_path}.pkl', 'rb') as f:
                                model = pickle.load(f)
                        else:
                            print(f"Training {base_learner}")
                            model.fit(features, target, sample_weight=self.sample_weights)
                            with open(f'{save_path}.pkl', 'wb') as f:
                                pickle.dump(model, f)

                    self.models.append(model)
                    # make predictions
                    predictions = model.predict(features)
                    # calculate the error using mean squared error
                    err = np.sum(self.sample_weights * (predictions - target)) / np.sum(self.sample_weights)
                    # calculate the model weight alpha
                    alpha = self.learning_rate * 0.5 * np.log((1 - err) / (err + 1e-10))
                    # update the aggregated predictions
                    agg_predictions += alpha * predictions
                    # update the sample weights
                
                