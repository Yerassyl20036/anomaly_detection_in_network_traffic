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

    # Uncomment the model training section
    # 2. Instantiate models
    # knn = KNNModel(n_neighbors=5)
    # mlp = MLPModel(hidden_layer_sizes=(64, 32), max_iter=300)
    # rf = RFModel(n_estimators=100)
    
    # # 3. Train models
    # knn.train(X_train, y_train)
    # mlp.train(X_train, y_train)
    # rf.train(X_train, y_train)
    
    # # 4. Predict
    # y_pred_knn = knn.predict(X_test)
    # y_pred_mlp = mlp.predict(X_test)
    # y_pred_rf  = rf.predict(X_test)
    
    # # 5. Evaluate & Visualize
    # print("=== KNN Results ===")
    # plot_confusion_matrix(y_test, y_pred_knn, title='KNN Confusion Matrix')
    # print_classification_report(y_test, y_pred_knn)
    
    # print("=== MLP Results ===")
    # plot_confusion_matrix(y_test, y_pred_mlp, title='MLP Confusion Matrix')
    # print_classification_report(y_test, y_pred_mlp)
    
    # print("=== Random Forest Results ===")
    # plot_confusion_matrix(y_test, y_pred_rf, title='RF Confusion Matrix')
    # print_classification_report(y_test, y_pred_rf)
    
    # # If you want ROC/PR curves (binary case), you need predict_proba
    # # Convert y_test to 0/1 if your data is labeled differently
    # # Example: "anomaly" => 1, "normal" => 0
    # # Make sure this aligns with your dataset

    # # Example: let’s assume "anomaly" == 1, "normal" == 0
    # # and we do a small conversion function:
    # def label_to_binary(y):
    #     return [1 if label == 'anomaly' else 0 for label in y]

    # y_test_bin = label_to_binary(y_test)

    # # kNN proba
    # y_proba_knn = knn.predict_proba(X_test)
    # if y_proba_knn is not None:
    #     # Probability of the "anomaly" class (index=1 if classes_ = [0,1])
    #     y_scores_knn = y_proba_knn[:, 1]
    #     plot_roc_curve(y_test_bin, y_scores_knn, label='kNN')
    #     plot_precision_recall_curve(y_test_bin, y_scores_knn, label='kNN')
    
    # # MLP proba
    # y_proba_mlp = mlp.predict_proba(X_test)
    # y_scores_mlp = y_proba_mlp[:, 1]
    # plot_roc_curve(y_test_bin, y_scores_mlp, label='MLP')
    # plot_precision_recall_curve(y_test_bin, y_scores_mlp, label='MLP')
    
    # # RF proba
    # y_proba_rf = rf.predict_proba(X_test)
    # y_scores_rf = y_proba_rf[:, 1]
    # plot_roc_curve(y_test_bin, y_scores_rf, label='Random Forest')
    # plot_precision_recall_curve(y_test_bin, y_scores_rf, label='Random Forest')

if __name__ == "__main__":
    main()