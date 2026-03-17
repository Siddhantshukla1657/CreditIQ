"""
model.py — Data Pipeline + Model Training for Loan Default Predictor
Loads the Kaggle Credit Risk Dataset, preprocesses it, trains Logistic Regression
and Random Forest classifiers, evaluates them, and saves the artifacts.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(filepath='data/credit_risk_dataset.csv'):
    """Load dataset, handle missing values, remove outliers."""
    df = pd.read_csv(filepath)

    # --- Remove extreme outlier ages (e.g., 123, 144) ---
    df = df[df['person_age'] <= 100].copy()

    # --- Remove extreme employment lengths (e.g., 123 years) ---
    df = df[df['person_emp_length'] <= 60].copy()

    # --- Fill missing values with median ---
    df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
    df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

    return df


def encode_features(df):
    """Encode categorical variables for ML."""
    df_encoded = df.copy()

    # Label encode loan_grade: A=1, B=2, ..., G=7
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df_encoded['loan_grade'] = df_encoded['loan_grade'].map(grade_map)

    # Binary encode cb_person_default_on_file: Y=1, N=0
    df_encoded['cb_person_default_on_file'] = df_encoded['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

    # One-hot encode loan_intent and person_home_ownership
    df_encoded = pd.get_dummies(df_encoded, columns=['loan_intent', 'person_home_ownership'], drop_first=True)

    return df_encoded


def prepare_data(df_encoded):
    """Split features/target and do train/test split + scaling."""
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_models(X_train, y_train):
    """Train Logistic Regression and Random Forest."""
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return lr, rf


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Evaluate a model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return metrics


def save_artifacts(lr, rf, scaler, feature_names, lr_metrics, rf_metrics, y_test):
    """Save models, scaler, and metadata."""
    os.makedirs('assets', exist_ok=True)

    joblib.dump(lr, 'assets/lr_model.pkl')
    joblib.dump(rf, 'assets/rf_model.pkl')
    joblib.dump(scaler, 'assets/scaler.pkl')
    joblib.dump(feature_names, 'assets/feature_names.pkl')

    # Save metrics for the app to load
    metrics_data = {
        'lr': {k: v for k, v in lr_metrics.items() if k not in ['y_pred', 'y_proba', 'fpr', 'tpr', 'confusion_matrix']},
        'rf': {k: v for k, v in rf_metrics.items() if k not in ['y_pred', 'y_proba', 'fpr', 'tpr', 'confusion_matrix']},
        'lr_cm': lr_metrics['confusion_matrix'].tolist(),
        'rf_cm': rf_metrics['confusion_matrix'].tolist(),
        'lr_fpr': lr_metrics['fpr'].tolist(),
        'lr_tpr': lr_metrics['tpr'].tolist(),
        'rf_fpr': rf_metrics['fpr'].tolist(),
        'rf_tpr': rf_metrics['tpr'].tolist(),
        'feature_names': feature_names,
        'rf_feature_importance': rf.feature_importances_.tolist(),
        'y_test': y_test.tolist(),
        'lr_y_proba': lr_metrics['y_proba'].tolist(),
        'rf_y_proba': rf_metrics['y_proba'].tolist(),
    }
    joblib.dump(metrics_data, 'assets/metrics_data.pkl')

    print("\n✅ All artifacts saved to assets/")


def main():
    print("🔄 Loading and cleaning dataset...")
    df = load_and_clean_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Default rate: {df['loan_status'].mean()*100:.1f}%")
    print(f"   Class distribution:\n{df['loan_status'].value_counts()}")

    print("\n🔄 Encoding features...")
    df_encoded = encode_features(df)
    print(f"   Encoded shape: {df_encoded.shape}")

    print("\n🔄 Preparing train/test split...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df_encoded)
    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("\n🔄 Training Logistic Regression...")
    lr, rf = train_models(X_train, y_train)

    print("\n📊 Evaluating models...")
    lr_metrics = evaluate_model(lr, X_test, y_test, 'Logistic Regression')
    rf_metrics = evaluate_model(rf, X_test, y_test, 'Random Forest')

    # Determine best model
    best = 'Random Forest' if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else 'Logistic Regression'
    print(f"\n🏆 Best Model: {best}")

    print("\n💾 Saving artifacts...")
    save_artifacts(lr, rf, scaler, feature_names, lr_metrics, rf_metrics, y_test)

    print("\n✅ Model training complete!")


if __name__ == '__main__':
    main()
