from models.pridict import train_pridict

train_pridict('pd-pd-hek293t-pe2.csv', lr=0.01, batch_size=2048, epochs=200, patience=30, num_runs=5, adjustment='none', num_features=24)