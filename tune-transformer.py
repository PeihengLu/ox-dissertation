from models.transformer import tune_transformer

tune_transformer('transformer-pd-hek293t-pe2.csv', percentage=1, dropout=0.1, lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=3, num_features=24)