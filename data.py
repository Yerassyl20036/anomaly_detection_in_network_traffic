# data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath="NSL-KDD-Dataset.csv", test_size=0.2, random_state=42):
    """
    1. Load NSL-KDD data from CSV.
    2. Encode categorical fields.
    3. Scale numeric fields.
    4. Split into train/test.
    """

    # 1. Load data
    df = pd.read_csv(filepath)
    
    # Example columns (names will differ based on your CSV)
    # Suppose "class" is your target: 'normal' or 'anomaly' etc.
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]

    X = df[feature_cols]
    y = df[target_col]

    # 2. Distinguish numeric vs categorical
    # Example: let's assume these columns are categorical
    categorical_cols = ["protocol_type", "service", "flag"]  # Adjust as necessary

    # Label encode categoricals
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # 3. Scale numeric features
    # We'll do a simple standard scaling on everything for simplicity
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test