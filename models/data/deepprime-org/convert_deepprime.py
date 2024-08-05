# extract the deepprime tables from the deepPrimeEditor xls file
# and save them as csv files
import sys

sys.path.append('../../../')

import pandas as pd
import os
import numpy as np

from utils.data_utils import convert_from_deepprime_org

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
    pe_version = row['PE system']
    
    # convert to lowercase
    cell_line = cell_line.lower()
    pe_version = pe_version.lower()
    
    cell_line = cell_line.replace('-', '_')
    pe_version = pe_version.replace('-', '_')
    
    # load the table
    df = pd.read_excel(xls, sheetname, skiprows=3)
    print(sheetname)
    print(df.head())
    if libary == 'Library-Small':
        convert_from_deepprime_org(df, cell_line, pe_version, 'small')
    else:
        convert_from_deepprime_org(df, cell_line, pe_version)