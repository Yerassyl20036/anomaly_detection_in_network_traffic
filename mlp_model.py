# mlp_model.py

from sklearn.neural_network import MLPClassifier

class MLPModel:
    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.0001, random_state=42):
        """
        hidden_layer_sizes: tuple for layer sizes, e.g. (64, 32)
        solver='sgd' to use Stochastic Gradient Descent
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            solver='sgd',
            learning_rate_init=0.01,
            max_iter=max_iter,
            alpha=alpha,
            random_state=random_state
        )
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)