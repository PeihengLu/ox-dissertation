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
from torch import nn

import copy
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

def clones(module, N):
    '''
    Produce N identical layers.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class StackedTransformer(nn.Module):
    def __init__(self, blocks):
        super(StackedTransformer, self).__init__()
        self.blocks = blocks

    def forward(self, X_k, X_v, X_q):
        for block in self.blocks:
            x = block(X_k, X_v, X_q)
            X_k = x
            X_v = x
            X_q = x
        return x