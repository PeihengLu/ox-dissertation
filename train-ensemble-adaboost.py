from models.ensemble_adaboost import EnsembleAdaBoost
from glob import glob
from os.path import basename, join as pjoin

model = EnsembleAdaBoost()

# for data in ['ensemble-dp-hek293t-pe2.csv','ensemble-pd-adv-pe2.csv', 'ensemble-pd-k562-pe2.csv', 'ensemble-pd-k562mlh1dn-pe2.csv',  'ensemble-pd-hek293t-pe2.csv']:
#     model.fit(data)

for data in glob(pjoin('models', 'data', 'ensemble', '*small*.csv')):
    model.fit(basename(data), fine_tune=True)