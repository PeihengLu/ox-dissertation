'''
Helper functions to facilitate training and inference of machine learning models
using pytorch, skorch and sklearn.
'''
import os
import numpy as np
import pandas as pd
import sklearn
import skorch
import torch

from typing import Tuple

# def cv_train(X_train: torch.Tensor, y_train: torch.Tensor, model: skorch.NeuralNetRegressor, model_name: str)

seed = 42

def undersample(features: pd.DataFrame, target:pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    ''' 
    sample 10% of the training data with editing efficiency of less than 10
    '''
    indices = target[target < 5].index
    # sample 10% of the data
    np.random.seed(seed)
    
    indices = np.random.choice(indices, int(0.9 * len(indices)), replace=False)
    target = target.drop(indices)
    features = features.drop(indices)
    
    return features, target