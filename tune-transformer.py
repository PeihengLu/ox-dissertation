from models.transformer import tune_transformer

tune_transformer('transformer-pd-hek293t-pe2.csv', num_features=24, num_runs=5, percentage=0.2, patience=20)