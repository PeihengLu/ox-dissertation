'''
Testing DeepPrime on the clinvar dataset
'''
from os.path import join as pjoin, basename

from models.deepprime import DeepPrime, preprocess_deep_prime, train_deep_prime, predict_deep_prime

data = 'dp-dp_transformer-hek293t-pe2.csv'
predict_deep_prime(data, hidden_size=128, num_layers=1, dropout=0, num_features=24, adjustment='none', source='transformer')