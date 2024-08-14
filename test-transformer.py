from models.transformer import predict_transformer

predict_transformer('transformer-dp-hek293t-pe2.csv', num_features=24, stack_dim=1, join_option='concat')