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

    # find the edit location
    edit_loc = df['lha-location-r'].values
        
    # align the sequence data at 20bp before the edit location
    wt_seq = [seq[max(0, loc-20):] for seq, loc in zip(wt_seq, edit_loc)]
    mut_seq = [seq[max(0, loc-20):] for seq, loc in zip(mut_seq, edit_loc)]

    # pad the sequence preedit to 20 if less
    wt_seq = ['N'*(max(20-loc, 0)) + seq for seq, loc in zip(wt_seq, edit_loc)]
    mut_seq = ['N'*(max(20-loc, 0)) + seq for seq, loc in zip(mut_seq, edit_loc)]

    # cap the sequence length to 50
    wt_seq = [seq[:50] for seq in wt_seq]
    mut_seq = [seq[:50] for seq in mut_seq]


    # find the ml data
    ml_data_fname = pjoin('..', 'conventional-ml', f"ml-{('-'.join(f.split('-')[1:]))}")
    # read the ml data if exists
    if isfile(ml_data_fname):
        ml_data = pd.read_csv(ml_data_fname)
    else:
        # convert the data to conventional ml
        ml_data = convert_to_conventional_ml(df)
        
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