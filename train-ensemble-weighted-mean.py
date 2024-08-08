from os.path import join as pjoin, isfile
import pandas as pd
import numpy as np
import sys
import collections
from utils.stats_utils import get_pearson_and_spearman_correlation
from utils.data_utils import k_fold_cross_validation_split
from models.conventional_ml_models import lasso_regression, ridge_regression, mlp, xgboost, random_forest
from sklearn.preprocessing import StandardScaler
import scipy
import pickle
import torch

import skorch
from models.conventional_ml_models import MLP
import os
import time

from models.ensemble_weighted_mean import WeightedMeanSkorch

start = time.time()

data = 'ml-pd-hek293t-pe2.csv'

dataset = pd.read_csv(pjoin('models', 'data', 'conventional-ml', data))
cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
data_source = '-'.join(data.split('-')[1:]).split('.')[0]

# 5 fold cross validation
fold = 5

features = dataset.iloc[:, :24].values
target = dataset.iloc[:, -2].values

print('Data loaded in {:.2f} seconds'.format(time.time() - start))
start = time.time()

for i in range(0, fold):
    print(f'Fold {i+1} of {fold}')
    train_features = features[dataset['fold'] != i]
    train_target = target[dataset['fold'] != i]
    
    # ============================
    # Lasso
    # ============================
    if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'lasso-{data_source}-fold-{i+1}.pkl')):
        print('Training Lasso')
        lasso_model = lasso_regression(train_features, train_target)
        lasso_model.fit(train_features, train_target)
        # pickle the model
        pickle.dump(lasso_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'lasso-{data_source}-fold-{i+1}.pkl'), 'wb'))
        print('Lasso trained in {:.2f} seconds'.format(time.time() - start))
    else:
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'lasso-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
            lasso_model = pickle.load(f)
    
    start = time.time()

    # ============================
    # Ridge
    # ============================
    if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'ridge-{data_source}-fold-{i+1}.pkl')):
        print('Training Ridge')
        ridge_model = ridge_regression(train_features, train_target)
        ridge_model.fit(train_features, train_target)
        # pickle the model
        pickle.dump(ridge_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'ridge-{data_source}-fold-{i+1}.pkl'), 'wb'))
        print('Ridge trained in {:.2f} seconds'.format(time.time() - start))
    else:
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'ridge-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
            ridge_model = pickle.load(f)
    
    # ============================
    # xgboost
    # ============================
    print('Training XGBoost')
    if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl')):
        xgboost_model = xgboost(train_features, train_target)
        xgboost_model.fit(train_features, train_target)
        # pickle the model
        pickle.dump(xgboost_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'wb'))
    else:
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
            xgboost_model = pickle.load(f)
    
    # ============================
    # random forest
    # ============================
    print('Training Random Forest')
    if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl')):
        rf_model = random_forest(train_features, train_target)
        rf_model.fit(train_features, train_target)
        # pickle the model
        pickle.dump(rf_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'wb'))
    else:
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
    
    # ============================
    # MLP
    # ============================
    
    # convert features and target to float32
    train_features = train_features.astype(np.float32)
    train_target = train_target.astype(np.float32)
    # add a dimension to the target
    train_target = train_target[:, np.newaxis]
    
    if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}.pkl')):
        print('Training MLP')
        mlp_model = skorch.NeuralNetRegressor(
            module=MLP,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            max_epochs=100,
            module__activation='relu',
            lr=0.005,
            device='cuda',
            batch_size=2048,
            train_split=skorch.dataset.ValidSplit(cv=5),
            module__hidden_layer_sizes = (64, 64,),
            # early stopping
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=15),
                skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer.pkl'), f_history=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history.json'), f_params=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}.pkl'), event_name='event_cp'),
                skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=10, T_mult=1),
            ]
        )
        mlp_model.fit(train_features, train_target)
    else:
        mlp_model = skorch.NeuralNetRegressor(
            module=MLP,
            device='cuda',
            module__hidden_layer_sizes = (64, 64,),
        )
        mlp_model.initialize()
        mlp_model.load_params(f_params=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}.pkl'), f_optimizer=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer.pkl'), f_history=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history.json'))
        
    # get the predictions
    ensemble_train = {
        'lasso': lasso_model.predict(train_features).flatten(),
        'ridge': ridge_model.predict(train_features).flatten(),
        'xgboost': xgboost_model.predict(train_features).flatten(),
        'random_forest': rf_model.predict(train_features).flatten(),
        'mlp': mlp_model.predict(train_features).flatten(),
    }
    ensemble_train = pd.DataFrame(ensemble_train, columns=['lasso', 'ridge', 'xgboost', 'random_forest', 'mlp'], dtype=np.float32)
    ensemble_train = ensemble_train.values.astype(np.float32)
    
    ensemble_model = WeightedMeanSkorch(n_regressors=5, save_path=pjoin('models', 'trained-models', 'ensemble', f'ensemble-{data_source}-fold-{i+1}'))
    ensemble_target = train_target
    
    # fit the ensemble model
    print('Training the ensemble model')
    print(ensemble_train.dtype)
    ensemble_model.fit(ensemble_train, ensemble_target)
    # print performance metrics
    ensemble_predictions = ensemble_model.predict(ensemble_train)
    ensemble_predictions = ensemble_predictions.flatten()
    ensemble_target = ensemble_target.flatten()
    pearson, spearman = get_pearson_and_spearman_correlation(ensemble_target, ensemble_predictions)
    
    print(f'Pearson\'s R: {pearson}')
    print(f'Spearman\'s correlation: {spearman}')