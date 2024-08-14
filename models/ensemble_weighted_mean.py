"""
Weighted mean implemented using a single layer neural network.
"""
import torch
import skorch

class WeightedMeanModel(torch.nn.Module):
    def __init__(self, n_regressors: int):
        """initialize the model

        Args:
            n_regressors (int): number of regressors in the ensemble
        """
        super(WeightedMeanModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=n_regressors, out_features=1, bias=False)

    def forward(self, X):
        return self.linear(X)
    
class WeightedMeanSkorch():
    def __init__(self, n_regressors: int, save_path: str = None):
        """initialize the model

        Args:
            n_regressors (int): number of regressors in the ensemble
        """
        self.model = skorch.NeuralNetRegressor(
            module=WeightedMeanModel,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            max_epochs=500,
            module__n_regressors=n_regressors,
            lr=0.05,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=2048,
            train_split=skorch.dataset.ValidSplit(cv=5),
            # early stopping
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=50),
                skorch.callbacks.Checkpoint(monitor='valid_loss_best', f_pickle=None, f_criterion=None, f_optimizer=f'{save_path}-optimizer.pkl', f_history=f'{save_path}-history.json', f_params=f'{save_path}-params.pkl', event_name='event_cp'),
                # skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, monitor='valid_loss', T_0=20, T_mult=1),
            ]
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_params(f_params=path)

    def load(self, save_path: str):
        self.model.load_params(f_params=f'{save_path}-params.pkl', f_optimizer=f'{save_path}-optimizer.pkl', f_history=f'{save_path}-history.json')
        
    def initialize(self):
        self.model.initialize()