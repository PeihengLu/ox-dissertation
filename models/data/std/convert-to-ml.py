# convert std data to conventional ml format
import sys
sys.path.append('../../../')
from utils.data_utils import convert_to_conventional_ml
from glob import glob
from os.path import join as pjoin

for std_data_source in glob(pjoin('*.csv')):
    if len(std_data_source.split('-')) > 4:
        continue
    print(std_data_source)
    convert_to_conventional_ml(std_data_source)