# visualize.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

def plot_roc_curve(y_true, y_scores, label='Model'):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, y_scores, label='Model'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores, pos_label=1)
    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
def plot_anomaly_scores(y_true, y_scores, label='Model'):
    plt.figure(figsize=(12, 8))
    plt.hist(y_scores[y_true == 0], bins=50, alpha=0.5, label='Normal')
    plt.hist(y_scores[y_true == 1], bins=50, alpha=0.5, label='Anomaly')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Anomaly Scores Distribution ({label})')
    plt.legend()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Feature importance for Random Forest
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(min(top_n, len(feature_names))), 
            importances[indices][:top_n],
            align='center')
    plt.xticks(range(min(top_n, len(feature_names))), 
               [feature_names[i] for i in indices][:top_n], 
               rotation=90)
    plt.tight_layout()
    plt.show()

def compare_models_roc(models_dict, X_test, y_test):
    """
    Compare multiple models on the same ROC curve
    """
    plt.figure(figsize=(12, 8))
    
    for name, model in models_dict.items():
        y_scores = model.predict_proba(X_test)[:, 1]
            
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
def plot_learning_curve(model, X, y, cv=10):
    """
    Plot learning curve for a model to check for overfitting/underfitting
    """
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='f1', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(12, 8))
    plt.title('Learning Curve')
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="red", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()