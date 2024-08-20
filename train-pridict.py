from models.pridict import train_pridict

lr = 0.005
batch_size = 2048
epochs = 500
patience = 20
num_runs = 2
num_features = 24

# train_pridict('pd-pd-hek293t-pe2.csv', lr=lr, batch_size=2048, epochs=500, patience=20, num_runs=3, num_features=24)
# train_pridict('pd-pd-adv-pe2.csv', lr=0.005, batch_size=2048, epochs=500, patience=20, num_runs=3, num_features=24)
# train_pridict('pd-pd-k562-pe2.csv', lr=0.005, batch_size=2048, epochs=500, patience=20, num_runs=3, num_features=24)
train_pridict('pd-dp-hek293t-pe2.csv', lr=0.0025, batch_size=1024, epochs=500, patience=20, num_runs=1, num_features=24)
# train_pridict('pd-pd-k562mlh1dn-pe2.csv', lr=0.005, batch_size=1024, epochs=500, patience=20, num_runs=3, num_features=24)