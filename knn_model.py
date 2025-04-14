# knn_model.py

from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            # Some classifiers don't have predict_proba
            return None