from models.transformer import train_transformer as train_transformer

from models.transformer_only import train_transformer as train_transformer_only

import torch, gc

# 'dp-pd-adv-pe2.csv', 'dp-pd-k562-pe2.csv', 'dp-pd-k562mlh1dn-pe2.csv', 'dp-pd-hek293t-pe2.csv', 'dp-dp-hek293t-pe2.csv'
files = ['transformer-pd-k562-pe2.csv']

for fname in files:
    train_transformer(fname, lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=3, num_features=24, percentage=1, dropout=0.2, num_encoder_units=1, annot=True, onehot=True)

    gc.collect()
