'''
Conventional ML models to be used as baselines
'''

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator

# ================================================
# Random Forest
# ================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def random_forest(X_train, y_train) -> BaseEstimator:
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

def support_vector_machine(X_train, y_train) -> BaseEstimator:
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

def xgboost(X_train, y_train) -> BaseEstimator:
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

def ridge_regression(X_train, y_train) -> Tuple[float, float]:
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

def lasso_regression(X_train, y_train) -> BaseEstimator:
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

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for i in range(len(self.hidden_layer_sizes)-1):
            x = self.activation(getattr(self, f'hidden_layer_{i}')(x))
        x = self.output_layer(x)
        return x

def mlp(X_train, y_train) -> BaseEstimator:
    '''
    MLP Regressor
    '''
    # Define the model
    mlp = skorch.NeuralNetRegressor(
        module=MLP,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        max_epochs=100,
        lr=0.005,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        module__hidden_layer_sizes = (64, 64,),
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