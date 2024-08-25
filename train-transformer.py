from models.transformer import train_transformer as train_transformer

from models.transformer_only import train_transformer as train_transformer_only

import torch, gc

# train_transformer('transformer-dp-hek293t-pe2.csv', lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=1, num_features=24, percentage=1, dropout=0.1, num_encoder_units=1, annot=True, onehot=True)

# torch.cuda.empty_cache()
# gc.collect()

# train_transformer('transformer-dp-hek293t-pe2.csv', lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=1, num_features=24, percentage=1, dropout=0.1, num_encoder_units=3, annot=True, onehot=True)

# torch.cuda.empty_cache()
# gc.collect()

train_transformer_only('transformer-dp-hek293t-pe2.csv', lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=3, num_features=24, percentage=0.1, dropout=0.1, num_encoder_units=5, annot=True, onehot=True, local=True)

