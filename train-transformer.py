from models.transformer import train_transformer

train_transformer('transformer-dp-hek293t-pe2.csv', lr=0.0025, batch_size=512, epochs=500, patience=20, num_runs=3, num_features=24, percentage=1, dropout=0.1, num_encoder_units=3)