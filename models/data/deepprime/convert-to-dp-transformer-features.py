import sys

sys.path.append('../../../')

import pandas as pd
import os
import numpy as np
import tqdm
from glob import glob

from utils.data_utils import k_fold_cross_validation_split


# export the tables to csv files+
for f in glob('*.csv'):
    data = pd.read_csv(f)
    print(data.head())
    
    # read the wt and mut sequences
    wt = data['wt-sequence']
    mut = data['mut-sequence']
    
    # load the corresponding ml data
    ml_data_fname = os.path.join('..', 'conventional-ml', f"ml-{('-'.join(f.split('-')[1:]))}")
    
    # load the ml data
    ml_data = pd.read_csv(ml_data_fname)
    
    # concatenate the sequence data with the ml data
    ml_data['wt-sequence'] = wt
    ml_data['mut-sequence'] = mut
    
    # move the sequence data to the first columns
    cols = ml_data.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    ml_data = ml_data[cols]
    
    # save the data
    target_fname = os.path.join('..', 'deepprime-transformer-features' , f"dp-{f.split('-')[1]}_transformer-{('-'.join(f.split('-')[2:]))}")

    ml_data.to_csv(target_fname, index=False)
    print(f"Saved to {target_fname}")
    