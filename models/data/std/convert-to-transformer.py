# padd the sequence length to 99 if less
# concatenate the sequence data with the ml data
import pandas as pd
import numpy as np
from os.path import join as pjoin, isfile
from glob import glob
import sys

sys.path.append('../../../')

from utils.data_utils import convert_to_conventional_ml

for f in glob("*.csv"):
    if len(f.split("-")) > 4: continue # special data for shap analysis
    print(f)
    target_fname = pjoin('..', 'transformer', f"transformer-{('-'.join(f.split('-')[1:]))}")
    print(target_fname)
    # read the data
    df = pd.read_csv(f)
    # get the sequence data
    wt_seq = df['wt-sequence'].values
    mut_seq = df['mut-sequence'].values
    
    # check protospacer location
    protospacer_location = df['protospacer-location-l'].values[0]
    
    if protospacer_location < 10:
        # padd the sequence data
        wt_seq = 'N' * (10 - protospacer_location) + wt_seq
        mut_seq = 'N' * (10 - protospacer_location) + mut_seq
    else:
        # remove the extra data
        wt_seq = wt_seq[protospacer_location - 10:]
        mut_seq = mut_seq[protospacer_location - 10:]
        
    # padd the sequence data if less than 99
    wt_seq = [s.ljust(99, 'N') for s in wt_seq]
    mut_seq = [s.ljust(99, 'N') for s in mut_seq]
    
    # find the ml data
    ml_data_fname = pjoin('..', 'conventional-ml', f"ml-{('-'.join(f.split('-')[1:]))}")
    # read the ml data if exists
    if isfile(ml_data_fname):
        ml_data = pd.read_csv(ml_data_fname)
    else:
        # convert the data to conventional ml
        ml_data = convert_to_conventional_ml(df, wt_seq, mut_seq)
        
    # concatenate the sequence data with the ml data
    ml_data['wt-sequence'] = wt_seq
    ml_data['mut-sequence'] = mut_seq
    
    # move the sequence data to the first columns
    cols = ml_data.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    ml_data = ml_data[cols]
    
    # concatenate positional information
    positions = ['protospacer-location-l','protospacer-location-r','pbs-location-l','pbs-location-r','rtt-location-wt-l','rtt-location-wt-r','rtt-location-mut-l','rtt-location-mut-r']
    # load the positional data from std data into the transformer data
    for pos in positions:
        ml_data[pos] = df[pos]
        
    # move group-id,editing-efficiency,fold to the last columns
    cols = ml_data.columns.tolist()
    cols.remove('group-id')
    cols.remove('editing-efficiency')
    cols.remove('fold')
    cols = cols + ['group-id','editing-efficiency','fold']
    ml_data = ml_data[cols]
    

    # save the data
    ml_data.to_csv(target_fname, index=False)