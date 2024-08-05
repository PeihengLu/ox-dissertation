'''
Convert the PRIDICT dataset to the std format
'''
import pandas as pd
import os
import sys

sys.path.append('../../../')

from utils.data_utils import convert_from_pridict2_org

# open the PRIDICT csv file
df = pd.read_csv('pridict2.csv')
convert_from_pridict2_org(df)