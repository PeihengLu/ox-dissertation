from models.ensemble_weighted_mean import EnsembleWeightedMean
from glob import glob
from os.path import join as pjoin, basename

model = EnsembleWeightedMean(optimization=False)

# for data in ['ensemble-pd-adv-pe2.csv', 'ensemble-pd-k562-pe2.csv', 'ensemble-pd-k562mlh1dn-pe2.csv',  'ensemble-pd-hek293t-pe2.csv']:
#     model.fit(data)

for data in glob(pjoin('models', 'data', 'ensemble', '*small*.csv')):
    model.fit(basename(data), fine_tune=True)