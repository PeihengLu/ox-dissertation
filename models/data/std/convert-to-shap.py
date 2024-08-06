# convert all std data to shap format
import sys
from os.path import join as pjoin, dirname, abspath, basename
from glob import glob

sys.path.append('../../../')
from utils.data_utils import convert_to_SHAP

for std_data_source in glob(pjoin('*.csv')):
    # if 'hek293' not in std_data_source or 'small' in std_data_source:
    #     continue
    print(std_data_source)
    convert_to_SHAP(std_data_source)