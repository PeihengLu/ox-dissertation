# convert all std data to pridict format
import sys
from os.path import join as pjoin, dirname, abspath, basename
from glob import glob

sys.path.append('../../../')

from utils.data_utils import convert_to_pridict

for std_data_source in glob(pjoin('*.csv')):
    if len(std_data_source.split('-')) > 4:
        continue
    # if 'pd' not in std_data_source:
    #     continue
    print(std_data_source)
    convert_to_pridict(std_data_source)