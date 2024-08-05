# convert all std data to deep prime format
import sys
from os.path import join as pjoin, dirname, abspath, basename
from glob import glob

sys.path.append('../../../')

from utils.data_utils import convert_to_deepprime

for std_data_source in glob('*.csv'):
    if len(std_data_source.split('-')) > 4:
        continue
    print(std_data_source)
    convert_to_deepprime(std_data_source)