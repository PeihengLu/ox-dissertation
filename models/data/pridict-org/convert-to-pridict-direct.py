'''
Convert the PRIDICT dataset to the pridict traning format
'''
import pandas as pd
import os
import sys

sys.path.append('../../../')

from utils.data_utils import convert_to_pridict_direct

# open the PRIDICT csv file
df = pd.read_csv('pridict2.csv')
convert_to_pridict_direct(df)