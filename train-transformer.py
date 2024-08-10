from models.transformer import train_transformer

train_transformer('transformer-pd-hek293t-pe2.csv', lr=0.005, batch_size=2048, epochs=100, patience=20, num_runs=5, num_features=24, percentage=0.2)