"""
train_model.py
Train a Random Forest classifier on the heart disease CSV and save the model.
Usage:
    python train_model.py --data path/to/heart_disease.csv --model_out path/to/heart_model.joblib
"""
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def build_and_train(X_train, y_train):
    # Pipeline with scaling and RandomForest
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def main(args):
    X, y = load_data(args.data)
    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_and_train(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, preds))
    if probs is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_test, probs))
        except:
            pass
    print("Classification report:\n", classification_report(y_test, preds))

    # Cross-validation (optional)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("5-fold CV accuracy: mean=%.3f std=%.3f" % (cv_scores.mean(), cv_scores.std()))

    joblib.dump(model, args.model_out)
    print(f"Model saved to {args.model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='heart_disease_sample.csv')
    parser.add_argument('--model_out', type=str, default='heart_model.joblib')
    args = parser.parse_args()
    main(args)
