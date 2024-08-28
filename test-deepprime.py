'''
Testing DeepPrime on the clinvar dataset
'''
from os.path import join as pjoin, basename

from models.deepprime import DeepPrime, preprocess_deep_prime, train_deep_prime, predict


for fname in ['dp-dp-hek293t-pe2.csv', 'dp-pd-adv-pe2.csv', 'dp-pd-k562-pe2.csv', 'dp-pd-k562mlh1dn-pe2.csv', 'dp-pd-hek293t-pe2.csv']:
    predict(fname, hidden_size=128, num_layers=1, dropout=0, num_features=24)