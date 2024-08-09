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

datas = ['ml-dp-hek293t-pe2.csv', 'ml-pd-hek293t-pe2.csv', 'ml-pd-adv-pe2.csv', 'ml-pd-k562-pe2.csv', 'ml-pd-k562mlh1dn-pe2.csv']
# 5 fold cross validation
fold = 5
num_runs = 3

for data in datas:
    dataset = pd.read_csv(pjoin('models', 'data', 'conventional-ml', data))
    cell_line = '-'.join(data.split('-')[1:3]).split('.')[0]
    data_source = '-'.join(data.split('-')[1:]).split('.')[0]


    features = dataset.iloc[:, :24].values
    target = dataset.iloc[:, -2].values

    print('Data loaded in {:.2f} seconds'.format(time.time() - start))
    start = time.time()

    for i in range(0, fold):
        print(f'Fold {i+1} of {fold}')
        train_features = features[dataset['fold'] != i]
        train_target = target[dataset['fold'] != i]
        
        test_features = features[dataset['fold'] == i]
        test_target = target[dataset['fold'] == i]
        
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
        
        start = time.time()
        # ============================
        # xgboost
        # ============================
        if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl')):
            print('Training XGBoost')
            xgboost_model = xgboost(train_features, train_target)
            xgboost_model.fit(train_features, train_target)
            # pickle the model
            pickle.dump(xgboost_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'wb'))
            print('XGBoost trained in {:.2f} seconds'.format(time.time() - start))
        else:
            with open(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                xgboost_model = pickle.load(f)
        
        start = time.time()
        # ============================
        # random forest
        # ============================
        if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl')):
            print('Training Random Forest')
            rf_model = random_forest(train_features, train_target)
            rf_model.fit(train_features, train_target)
            # pickle the model
            pickle.dump(rf_model, open(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'wb'))
            print('Random Forest trained in {:.2f} seconds'.format(time.time() - start))
        else:
            with open(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{i+1}.pkl'), 'rb') as f:
                rf_model = pickle.load(f)
        
        # ============================
        # MLP
        # ============================
        start = time.time()
        
        # convert features and target to float32
        train_features = train_features.astype(np.float32)
        train_target = train_target.astype(np.float32)
        # add a dimension to the target
        train_target = train_target[:, np.newaxis]
        
        if not isfile(pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer.pkl')):
            print('Training MLP')
            val_loss_best = np.inf
            for j in range(0, num_runs):
                mlp_model = skorch.NeuralNetRegressor(
                    module=MLP,
                    criterion=torch.nn.MSELoss,
                    optimizer=torch.optim.Adam,
                    max_epochs=400,
                    module__activation='relu',
                    lr=0.005,
                    device='cuda',
                    batch_size=2048,
                    train_split=skorch.dataset.ValidSplit(cv=5),
                    module__hidden_layer_sizes = (64, 64,),
                    # early stopping
                    callbacks=[
                        skorch.callbacks.EarlyStopping(patience=20),
                        skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer-tmp.pkl'), f_history=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history-tmp.json'), f_params=os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-tmp.pkl'), event_name='event_cp'),
                        skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=10, T_mult=1),
                    ]
                )
                mlp_model.initialize()
                mlp_model.fit(train_features, train_target)
                if np.min(mlp_model.history[:, 'valid_loss']) < val_loss_best:
                    val_loss_best = np.min(mlp_model.history[:, 'valid_loss'])
                    print(f'Run {i+1} of {num_runs} - Best validation loss: {val_loss_best}')
                    os.rename(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer-tmp.pkl'), os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer.pkl'))
                    os.rename(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history-tmp.json'), os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history.json'))
                    os.rename(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-tmp.pkl'), os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}.pkl'))
                else:
                    print(f'Run {i+1} of {num_runs} - Validation loss: {np.min(mlp_model.history[:, "valid_loss"])} not better than best validation loss: {val_loss_best}')
                    # remove the temporary files
                    os.remove(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer-tmp.pkl'))
                    os.remove(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history-tmp.json'))
                    os.remove(os.path.join('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-tmp.pkl'))
            print('MLP trained in {:.2f} seconds'.format(time.time() - start))
        else:
            mlp_model = skorch.NeuralNetRegressor(
                module=MLP,
                device='cuda',
                module__hidden_layer_sizes = (64, 64,),
            )
            mlp_model.initialize()
            mlp_model.load_params(f_params=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}.pkl'), f_optimizer=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-optimizer.pkl'), f_history=pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}-fold-{i+1}-history.json'))
        