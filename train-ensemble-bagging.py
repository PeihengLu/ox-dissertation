from models.ensemble_bagging import EnsembleBagging

model = EnsembleBagging()

for data in ['ensemble-pd-adv-pe2.csv', 'ensemble-pd-k562-pe2.csv', 'ensemble-pd-k562mlh1dn-pe2.csv', 'ensemble-dp-hek293t-pe2.csv', 'ensemble-pd-hek293t-pe2.csv']:
    model.fit(data)