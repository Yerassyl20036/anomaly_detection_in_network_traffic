# main.py

from data import load_and_preprocess_data
from knn_model import KNNModel
from mlp_model import MLPModel
from rf_model import RFModel
from visualize import (
    plot_confusion_matrix, 
    print_classification_report, 
    plot_roc_curve, 
    plot_precision_recall_curve
)
from sklearn.metrics import precision_recall_fscore_support

def performOnModel(model, X_train, y_train, X_test, y_test, model_name):
    
    # 3. Train models
    model.train(X_train, y_train)
    
    # 4. Predict
    y_pred = model.predict(X_test)
    
    # 5. Evaluate & Visualize
    print(f"=== {model_name} Results ===")
    plot_confusion_matrix(y_test, y_pred, title=f'{model_name} Confusion Matrix')
    print_classification_report(y_test, y_pred)
    
    # If you want ROC/PR curves (binary case), you need predict_proba
    # Convert y_test to 0/1 if your data is labeled differently
    # Example: "anomaly" => 1, "normal" => 0
    # Make sure this aligns with your dataset

    # Example: let’s assume "anomaly" == 1, "normal" == 0
    # and we do a small conversion function:
    def label_to_binary(y):
        return [1 if label == 'anomaly' else 0 for label in y]

    y_test_bin = label_to_binary(y_test)

    # kNN proba
    y_proba = model.predict_proba(X_test)
    if y_proba is not None:
        y_scores = y_proba[:, 1]

        plot_roc_curve(y_test_bin, y_scores, label=model_name)
        plot_precision_recall_curve(y_test_bin, y_scores, label=model_name)

def main():
    # 1. Load / preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Print shapes and sample data
    print("Data Shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    print("\nFeature Names:")
    print(X_train.columns)
    
    print("\nSample Data:")
    print("X_train head:")
    print(X_train.head())
    print("\ny_train head:")
    print(y_train.head())

    # 2. Instantiate models
    knn = KNNModel(n_neighbors=5)
    mlp = MLPModel(hidden_layer_sizes=(64, 32), max_iter=300)
    rf = RFModel(n_estimators=100)

    model_dic = {
        'KNN': knn, 
        'MLP': mlp, 
        'RF': rf
    }

    for model_name, model in model_dic.items():
        performOnModel(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_name=model_name)

if __name__ == "__main__":
    main()