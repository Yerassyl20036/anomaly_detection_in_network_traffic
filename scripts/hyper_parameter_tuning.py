#!/usr/bin/env python3
"""
hyperparameter_tuning.py

Grid-search hyperparameter tuning for all models in the project.
Uses same structure as main.py: load data, split, tune (accuracy), print best params & metrics, and save artifacts.
"""
import os, sys
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    fbeta_score
)

# add project root
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

BASE_OUT = script_dir.parent / 'generated'
BASE_OUT.mkdir(parents=True, exist_ok=True)

from data import load_and_preprocess_data
from knn_model import KNNModel
from mlp_model import MLPModel
from rf_model import RFModel

def perform_tuning(estimator, params, X, y):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y)
    return grid

def plot_param_sets(name, cv_results):
    out_dir = BASE_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # plot each parameter combination's mean CV score with numeric x-axis
    param_list = cv_results['params']
    scores = cv_results['mean_test_score']
    x = np.arange(len(param_list))

    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(x, scores)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90, fontsize=8)
    ax.set_title(f'{name}: hyperparameter combinations vs CV score')
    ax.set_xlabel('Combination index (see mapping file)')
    ax.set_ylabel('Mean CV score')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / 'param_sets_performance.png')
    plt.close(fig)

    # save mapping of index to parameter dict
    mapping_file = out_dir / 'param_map.txt'
    with open(mapping_file, 'w') as f:
        for idx, p in enumerate(param_list):
            f.write(f'{idx}: {p}')


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    param_grids = {
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 13, 17, 21, 30],
            'weights': ['uniform', 'distance']
        },
        'MLP': {
            'hidden_layer_sizes': [(32,), (64, 32), (16), (8)],
            'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-3, 1e-4],
            'max_iter': [300, 500],
            'batch_size':[32, 64, 128]
        },
        'RF': {
            'n_estimators': [50, 100],
            'max_depth': [2, 5, 10, 20],
            'min_samples_leaf': [1, 3],
            'min_samples_split': [2, 5]
        }
    }

    models = {
        'KNN': KNNModel().model,
        'MLP': MLPModel().model,
        'RF': RFModel().model
    }

    summary = {}
    for name, model in models.items():
        print(f"\n--- Tuning {name} ---")
        grid = perform_tuning(model, param_grids[name], X_train, y_train)
        best = grid.best_estimator_

        out_dir = BASE_OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, out_dir / f"{name}_best.pkl")
        joblib.dump(grid, out_dir / f"{name}_grid.pkl")

        plot_param_sets(name, grid.cv_results_)

        y_pred = best.predict(X_test)
        y_prob = None
        if hasattr(best, 'predict_proba'):
            y_prob = best.predict_proba(X_test)[:, 1]
        else:
            try:
                y_prob = best.decision_function(X_test)
            except:
                y_prob = None

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        pr = average_precision_score(y_test, y_prob) if y_prob is not None else None
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)

        summary[name] = {
            'accuracy': acc,
            'roc_auc': roc,
            'pr_auc': pr,
            'f1_score': f1,
            'mcc': mcc,
            'f2_score': f2
        }

    for name, m in summary.items():
        print(f"\n{name} results:")
        print(f" accuracy  = {m['accuracy']:.4f}")
        if m['roc_auc'] is not None:
            print(f" roc_auc   = {m['roc_auc']:.4f}")
            print(f" pr_auc    = {m['pr_auc']:.4f}")
        print(f" f1_score  = {m['f1_score']:.4f}")
        print(f" mcc       = {m['mcc']:.4f}")
        print(f" f2_score  = {m['f2_score']:.4f}")

if __name__ == '__main__':
    main()