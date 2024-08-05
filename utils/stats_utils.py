'''
Functions for preprocessing sequence data
'''
import numpy as np
import pandas as pd
import torch
import scipy

from typing import List, Tuple, Union


def get_pearson_and_spearman_correlation(target: Union[List[int], np.ndarray, pd.Series, torch.Tensor], output: Union[List[int], np.ndarray, pd.Series, torch.Tensor]) -> Tuple[float, float]:
    '''
    Returns the pearson's R and Spearman's correlation between the target and output
    '''
    target = np.array(target)
    output = np.array(output)

    pearson = np.corrcoef(target, output)[0, 1]
    spearman = scipy.stats.spearmanr(target, output)[0]
    
    return pearson, spearman