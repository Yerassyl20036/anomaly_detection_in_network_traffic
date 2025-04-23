# rf_model.py

from sklearn.ensemble import RandomForestClassifier

class RFModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, class_weight='balanced'):
        """
        n_estimators: number of trees
        max_depth: maximum depth of trees
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)