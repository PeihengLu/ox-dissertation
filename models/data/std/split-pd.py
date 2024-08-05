'''
split the pridict hek293t pe2 data based on editing types
'''
import pandas as pd
import numpy as np
import os
import sys

# load the data
data = pd.read_csv('std-pd-hek293t-pe2.csv')

data_replace = data[data['mut-type'] == 0]
data_insert = data[data['mut-type'] == 1]
data_delete = data[data['mut-type'] == 1]

# save the data
data_replace.to_csv('std-pd-hek293t-pe2-replace.csv', index=False)
data_insert.to_csv('std-pd-hek293t-pe2-insert.csv', index=False)
data_delete.to_csv('std-pd-hek293t-pe2-delete.csv', index=False)
