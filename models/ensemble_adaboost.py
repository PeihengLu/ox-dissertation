class EmsenbleAdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        
    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)
            error = np.sum(w * (y_pred != y))
            alpha = self.learning_rate * np.log((1 - error) / error)
            w = w * np.exp(alpha * (y_pred != y))
            w = w / np.sum(w)
            self.models.append(model)
            self.alphas.append(alpha)
            
    def predict(self, X):
        n = len(X)
        y_pred = np.zeros(n)
        for model, alpha in zip(self.models, self.alphas):
            y_pred += alpha * model.predict(X)
        return
    
    def save(self, path):
        