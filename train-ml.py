from models.conventional_ml_models import random_forest, xgboost, ridge_regression, mlp
from glob import glob
from os.path import join as pjoin, basename
import pandas as pd
import numpy as np
import pickle

for data in glob(pjoin('models', 'data', 'conventional-ml', '*small*.csv')):
    save_name = basename(data).split('.')[0]
    data_source = save_name.split('-')[1:]
    data_source = '-'.join(data_source)
    mlp_save_name = pjoin('models', 'trained-models', 'conventional-ml', f'mlp-{data_source}')   
    
    data = pd.read_csv(data)
    for fold in range(5):
        data_fold = data[data['fold'] != fold]
        features = data_fold.iloc[:, :24]
        targets = data_fold.iloc[:, -2]
        features = np.array(features).astype(np.float32)
        targets = np.array(targets).astype(np.float32)
        
        rf = random_forest()
        rf.fit(features, targets)
        
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'random_forest-{data_source}-fold-{fold+1}.pkl'), 'wb') as f:
            pickle.dump(rf, f)
            
        xgb = xgboost()
        xgb.fit(features, targets)
        
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'xgboost-{data_source}-fold-{fold+1}.pkl'), 'wb') as f:
            pickle.dump(xgb, f)
            
        ridge = ridge_regression()
        ridge.fit(features, targets)
        
        with open(pjoin('models', 'trained-models', 'conventional-ml', f'ridge_regression-{data_source}-fold-{fold+1}.pkl'), 'wb') as f:
            pickle.dump(ridge, f)
            
        mlp_model = mlp(mlp_save_name + f'-fold-{fold+1}')
        mlp_model.fit(features, targets)
        
    
        
        
        