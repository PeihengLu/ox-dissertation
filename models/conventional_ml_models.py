'''
Conventional ML models to be used as baselines
'''

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator
import skorch

# ================================================
# Random Forest
# ================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def random_forest() -> BaseEstimator:
    '''
    Random Forest Regressor
    '''
    # Define the model
    rf = RandomForestRegressor()

    # Define the hyperparameters
    # param_grid = {
    #     'n_estimators': [50, 100, 150],
    #     'max_depth': [5, 10, 15]
    # }

    # estimator = grid_search(X_train, y_train, rf, 'Random Forest', param_grid)
    
    estimator = rf.set_params(n_estimators=200, max_depth=10)

    return estimator

# ================================================
# Support Vector Machine
# ================================================
from sklearn.svm import SVR

def support_vector_machine() -> BaseEstimator:
    '''
    Support Vector Machine Regressor
    '''
    # Define the model
    svm = SVR()

    # Define the hyperparameters
    param_grid = {
        # regularization strength
        'C': [0.1, 1, 10, 100],
        # margin of tolerance
        'epsilon': [0.1, 0.2, 0.5, 1]
    }

    corr = grid_search(X_train, y_train, svm, 'Support Vector Machine', param_grid)

    return corr

# ================================================
# XGBoost
# ================================================
from xgboost import XGBRegressor

def xgboost() -> BaseEstimator:
    '''
    XGBoost Regressor
    '''
    # Define the model
    xgb = XGBRegressor()

    # Define the hyperparameters
    # param_grid = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'max_depth': [5, 10, 15, 20]
    # }

    # corr = grid_search(X_train, y_train, xgb, 'XGBoost', param_grid)
    
    corr = xgb.set_params(n_estimators=200, max_depth=5)

    return corr

# ================================================
# Ridge Regression
# ================================================
from sklearn.linear_model import Ridge

def ridge_regression() -> Tuple[float, float]:
    '''
    Ridge Regression
    '''
    # Define the model
    ridge = Ridge()

    # Define the hyperparameters
    # param_grid = {
    #     # regularization strength
    #     'alpha': np.logspace(-4, 1, 10)
    # }

    # corr = grid_search(X_train, y_train, ridge, 'Ridge Regression', param_grid)
    
    corr = ridge.set_params(alpha=494.17133613238286)

    return corr

# ================================================
# Lasso Regression
# ================================================
from sklearn.linear_model import Lasso

def lasso_regression() -> BaseEstimator:
    '''
    Lasso Regression
    '''
    # Define the model
    lasso = Lasso()

    # Define the hyperparameters
    # param_grid = {
    #     # regularization strength
    #     'alpha': np.logspace(-4, 4, 20)
    # }

    # estimator = grid_search(X_train, y_train, lasso, 'Lasso Regression', param_grid)
    
    estimator = lasso.set_params(alpha=0.006158482110660266)

    return estimator

# ================================================
# MLP
# ================================================
import torch, skorch
from sklearn.preprocessing import StandardScaler

class MLP(torch.nn.Module):
    def __init__(self, input_dim=24, hidden_layer_sizes=(100), activation='relu'):
        super(MLP, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes

        self.input_layer = torch.nn.Linear(input_dim, hidden_layer_sizes[0])
        for i, hidden_layer_size in enumerate(hidden_layer_sizes[1:]):
            setattr(self, f'hidden_layer_{i}', torch.nn.Linear(hidden_layer_sizes[i], hidden_layer_size))
        self.output_layer = torch.nn.Linear(hidden_layer_sizes[-1], 1)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh()
        elif activation == 'logistic':
            self.activation = torch.nn.Sigmoid()

    def forward(self, x, g=None, sample_weight=None):
        x = self.activation(self.input_layer(x))
        for i in range(len(self.hidden_layer_sizes)-1):
            x = self.activation(getattr(self, f'hidden_layer_{i}')(x))
        x = self.output_layer(x)
        return x

class MLPSkorch(skorch.NeuralNet):
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

def mlp(save_path, fine_tune: bool= False) -> BaseEstimator:
    '''
    MLP Regressor
    '''
    # Define the model
    mlp = skorch.NeuralNetRegressor(
        module=MLP,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=300,
        lr=0.005,
        batch_size=1024,
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        module__hidden_layer_sizes = (64, 64,),
        # early stopping
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=20),
            skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'{save_path}.pt', f_optimizer=None, f_history=None, f_pickle=None, f_criterion=None),
            skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=10, T_mult=1),
        ]
    )

    # # Define the hyperparameters
    # param_grid = {
    #     'module__hidden_layer_sizes':[(64,), (128,), (256,)],
    #     'module__activation': ['relu', 'tanh', 'logistic'],
    #     'lr': [0.1, 1, 10]
    # }

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # y_train = y_train.reshape(-1, 1)
    # y_train = scaler.fit_transform(y_train)

    # # make sure x train and y train are numpy arrays
    # X_train = X_train.astype(np.float32)
    # y_train = y_train.astype(np.float32)

    # # Grid search
    # estimator = grid_search(X_train, y_train, mlp, 'MLP', param_grid)
    
    estimator = mlp.set_params(module__hidden_layer_sizes=(64,64), module__activation='relu', lr=0.005)

    return estimator

def mlp_weighted(save_path: str, fine_tune: bool=False) -> BaseEstimator:
    '''
    MLP Regressor with sample weights
    '''
    # Define the model
    mlp = MLPSkorch(
        module=MLP,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=300,
        lr=0.005,
        batch_size=1024,
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        module__hidden_layer_sizes = (64, 64,),
        # early stopping
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=20),
            skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_params=f'{save_path}.pt', f_optimizer=None, f_history=None, f_pickle=None, f_criterion=None),
            skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=10, T_mult=1),
        ]
    )
    
    estimator = mlp.set_params(module__hidden_layer_sizes=(64,64), module__activation='relu', lr=0.005)

    return estimator

# ================================================
# Predictions
# ================================================

from os.path import join as pjoin, basename
import pickle

def mlp_predict(data: str) -> np.ndarray:
    """
    Make predictions using the MLP model on a csv file
    """
    data_source = '-'.join(basename(data).split('.')[0].split('-')[1:])
    data_path = pjoin('models', 'data', 'conventional-ml', f'ml-{data_source}.csv')
    
    data = pd.read_csv(data_path)
    data = data.dropna()
    # run predictions across five folds
    predictions = dict()
    for fold in range(5):
        fold_data = data[data['fold'] == fold]
        # load the model
        model = mlp(f'')
        model.initialize()
        model.load_params(f_params=f'models/trained-models/conventional-ml/mlp-{data_source}-fold-{fold+1}.pt')

        features = fold_data.iloc[:, :24].values
        features = features.astype(np.float32)
        
        # make predictions
        predictions[fold] = model.predict(features)

    return predictions

def xgboost_predict(data: str) -> np.ndarray:
    """
    Make predictions using the XGBoost model on a csv file
    """
    data_source = '-'.join(basename(data).split('.')[0].split('-')[1:])
    data_path = pjoin('models', 'data', 'conventional-ml', f'ml-{data_source}.csv')
    
    data = pd.read_csv(data_path).dropna()
    # run predictions across five folds
    predictions = dict()
    for fold in range(5):
        fold_data = data[data['fold'] == fold]
        # load the model
        with open(f'models/trained-models/conventional-ml/xgboost-{data_source}-fold-{fold+1}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        features = fold_data.iloc[:, :24].values
        features = features.astype(np.float32)
        
        # make predictions
        predictions[fold] = model.predict(features)

    return predictions

def random_forest_predict(data: str) -> np.ndarray:
    """
    Make predictions using the Random Forest model on a csv file
    """
    data_source = '-'.join(basename(data).split('.')[0].split('-')[1:])
    data_path = pjoin('models', 'data', 'conventional-ml', f'ml-{data_source}.csv')
    
    data = pd.read_csv(data_path).dropna()
    # run predictions across five folds
    predictions =dict()
    for fold in range(5):
        fold_data = data[data['fold'] == fold]
        # load the model
        with open(f'models/trained-models/conventional-ml/random_forest-{data_source}-fold-{fold+1}.pkl', 'rb') as f:
            model = pickle.load(f)

        features = fold_data.iloc[:, :24].values
        features = features.astype(np.float32)
        
        # make predictions
        predictions[fold] = model.predict(features)

    return predictions

def ridge_predict(data: str) -> np.ndarray:
    """
    Make predictions using the Ridge Regression model on a csv file
    """
    data_source = '-'.join(basename(data).split('.')[0].split('-')[1:])
    data_path = pjoin('models', 'data', 'conventional-ml', f'ml-{data_source}.csv')
    
    data = pd.read_csv(data_path).dropna()
    # run predictions across five folds
    predictions = dict()
    for fold in range(5):
        fold_data = data[data['fold'] == fold]
        # load the model
        with open(f'models/trained-models/conventional-ml/ridge-{data_source}-fold-{fold+1}.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # load the data
        data = pd.read_csv(data_path)
        features = fold_data.iloc[:, :24].values
        features = features.astype(np.float32)
        
        # make predictions
        predictions[fold] = model.predict(features)

    return predictions

# ================================================
# Helper functions
# ================================================
from utils.stats_utils import get_pearson_and_spearman_correlation
from sklearn.metrics import make_scorer


def grid_search(X_train, y_train, model, model_name, param_grid) -> BaseEstimator:
    '''
    Grid search for hyperparameters of the models
    '''
    # Grid search
    scorer = make_scorer(get_score, greater_is_better=True)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=scorer, n_jobs=4, verbose=10, error_score='raise')
    grid_search.fit(X_train, y_train)

    # Best hyperparameters
    best_params = grid_search.best_params_

    # print the best hyperparameters
    print(f"{model_name} Best hyperparameters: {best_params}")

    # Train the model with the best hyperparameters
    estimator = model.set_params(**best_params)

    return estimator

def get_score(y_true, y_pred):
    '''
    Calculate the score
    '''
    print(y_pred.shape, y_true.shape)
    if np.mean(y_pred) == y_pred[0] or np.mean(y_true) == y_true[0]:
        return 0  # Returning 0 in these cases, you can choose another appropriate value
    pearson, spearman = get_pearson_and_spearman_correlation(y_true, y_pred)
    return pearson