from models.transformer import train_transformer

train_transformer('transformer-pd-hek293t-pe2.csv', lr=0.001, batch_size=1024, epochs=500, patience=10, num_runs=3, num_features=24, percentage=1, dropout=0.3, stack_dim=1)