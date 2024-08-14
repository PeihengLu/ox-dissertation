import sys

sys.path.append('../../../')

import pandas as pd
import os
import numpy as np
import tqdm

from utils.data_utils import k_fold_cross_validation_split

# open the deepPrimeEditor xls file
xls = pd.ExcelFile('deepprime-org.xlsx')
# read the summary file
summary = pd.read_csv('summary.csv')

print(summary.head())

# export the tables to csv files+
for ind, row in summary.iterrows():
    sheetname = str(row['Index'])
    
    libary = row['Library']
    if 'off' in libary or 'Off' in libary:
        print('Skipping off-target library')
        continue
    cell_line = row['Cell line']
    editor = row['PE system']
    
    # convert to lowercase
    cell_line = cell_line.lower()
    editor = editor.lower()
    
    cell_line = cell_line.replace('-', '_')
    editor = editor.replace('-', '_')
    
    # load the table
    data = pd.read_excel(xls, sheetname, skiprows=3)
    
    print(data.head())

    # replace the '-' in editor and cell line with '_'
    cell_line = cell_line.lower()
    editor = editor.lower()
    cell_line = cell_line.replace('-', '_')
    editor = editor.replace('-', '_')

    output = []

    # result columns
    result_columns = ['cell-line', 'group-id', 'mut-type', 'edit-len', 'wt-sequence', 'mut-sequence', 'protospacer-location-l', 'protospacer-location-r', 'pbs-location-l', 'pbs-location-r', 'rtt-location-wt-l', 'rtt-location-wt-r', 'rtt-location-mut-l', 'rtt-location-mut-r', 'lha-location-l', 'lha-location-r', 'rha-location-wt-l', 'rha-location-wt-r', 'rha-location-mut-l', 'rha-location-mut-r', 'spcas9-score', 'editing-efficiency']

    g_id = 0
    prev = ""
    
    # rename wt and edited sequences
    data = data.rename(columns={"Wide target sequence (Target 74bps = 4bp neighboring sequence + 20 bp protospacer + 3 bp NGG + 47 bp neighboring sequence)": "wt-sequence"})
    data = data.rename(columns={"Wide target sequence\n(Target 74bps = 4bp neighboring sequence + 20 bp protospacer + 3 bp NGG + 47 bp neighboring sequence)": "wt-sequence"})

    data = data.rename(columns={"Edited target sequence (Target 74bps = RT-PBS corresponding region and masked by 'x')": "mut-sequence"})
    data = data.rename(columns={"Edited target sequence\n(Target 74bps = RT-PBS corresponding region and masked by 'x')": "mut-sequence"})
    
    # convert the x in the mut-sequence to N
    data['mut-sequence'] = data['mut-sequence'].apply(lambda x: x.replace('x', 'N'))
        
    print(data.columns)
    # drop columns 'Fold', 'Data set name', 'Trained model'
    data = data.drop(columns=['Fold', 'Data set name', 'Trained model'])    
    
    # add a group id column
    group_id = -1
    prev = ""
    group_ids = []

    # iterate over the data
    for ind, item in tqdm.tqdm(data.iterrows(), total=len(data)):
        if item['wt-sequence'] == prev:
            group_ids.append(group_id)
        else:
            group_id += 1
            group_ids.append(group_id)
            prev = item['wt-sequence']
    
    data['group-id'] = group_ids
    
    print(data.columns)
    # move group id to the second to last column
    cols = data.columns.tolist()
    cols = cols[:-2] + [cols[-1]] + [cols[-2]]
    data = data[cols]
    
    print(data.columns)
    
    # add the fold column
    data = k_fold_cross_validation_split(data, 5)
    
    # save the data
    if 'Small' in libary:
        data.to_csv(f'dp-dp_org_small-{cell_line}-{editor}.csv', index=False)
    else:
        data.to_csv(f'dp-dp_org-{cell_line}-{editor}.csv', index=False)