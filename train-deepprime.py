'''
Training DeepPrime
'''
from os.path import join as pjoin, basename

from models.deepprime import DeepPrime, preprocess_deep_prime, train_deep_prime


# org dp dataset
fname = 'dp-dp-hek293t-pe2.csv'
train_deep_prime(fname, hidden_size=128, num_features=24, num_layers=1, dropout=0.05, epochs=200, batch_size=1024, lr=0.005, patience=5, device='cuda', num_runs=5)

for fname in ['dp-pd-adv-pe2.csv', 'dp-pd-k562-pe2.csv', 'dp-pd-k562mlh1dn-pe2.csv', 'dp-dp-hek293t-pe2.csv']:
    train_deep_prime(fname, hidden_size=128, num_features=24, num_layers=1, dropout=0.05, epochs=200, batch_size=1024, lr=0.005, patience=20, device='cuda', num_runs=5)
    
