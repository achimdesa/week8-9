#Import necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def load_fraud_data(file_path):
    """Load the fraud data from a CSV file."""
    fraud_data = pd.read_csv(file_path)
    print("Fraud Data Sample:")
    display = print(fraud_data.head(), "\n")
    return fraud_data,display


def prepare_data(fraud_data, creditcard_data):
    """Prepare feature and target datasets for both fraud and credit card data."""
    # For Fraud_Data.csv
    fraud_data_encoded = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)
    
    # Define columns to drop
    columns_to_drop = ['class', 'signup_time', 'purchase_time', 'ip_address', 'ip_address_int', 'device_id']
    
    # Drop only the columns that exist
    columns_to_drop = [col for col in columns_to_drop if col in fraud_data_encoded.columns]
    
    X_fraud = fraud_data_encoded.drop(columns=columns_to_drop)
    y_fraud = fraud_data_encoded['class']

    # For creditcard.csv
    creditcard_data_encoded = pd.get_dummies(creditcard_data, drop_first=True)
    X_credit = creditcard_data_encoded.drop(columns=['Class'], errors='ignore')  # Ignore if 'Class' not found
    y_credit = creditcard_data_encoded['Class']

    return X_fraud, y_fraud, X_credit, y_credit

def split_data(X_fraud, y_fraud, X_credit, y_credit):
    """Split the datasets into training and testing sets."""
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
    )
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
        X_credit, y_credit, test_size=0.3, random_state=42, stratify=y_credit
    )
    return (X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud,
            X_train_credit, X_test_credit, y_train_credit, y_test_credit)
'''
def evaluate_model1(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }
'''
def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree model."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost model."""
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train):
    """Train a Multi-layer Perceptron model."""
    model = MLPClassifier(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_model(model_type, X_train, y_train):
    """
    Train a model based on the specified model type.
    
    Parameters:
    - model_type: Type of model to train (e.g., "logistic_regression", "decision_tree", etc.).
    - X_train: Training features.
    - y_train: Training labels.
    
    Returns:
    - model: Trained model.
    """
    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif model_type == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier()
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    print("All models are trained successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return performance metrics.
    
    Parameters:
    - model: Trained model.
    - X_test: Test features.
    - y_test: Test labels.
    
    Returns:
    - metrics: Dictionary of evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
        
    }
    return metrics

def train_and_evaluate_models(X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud,
                              X_train_credit, y_train_credit, X_test_credit, y_test_credit):
    """
    Train and evaluate multiple models for fraud and credit datasets.
    
    Parameters:
    - X_train_fraud, y_train_fraud: Training data for fraud dataset.
    - X_test_fraud, y_test_fraud: Test data for fraud dataset.
    - X_train_credit, y_train_credit: Training data for credit dataset.
    - X_test_credit, y_test_credit: Test data for credit dataset.
    
    Returns:
    - results_df: DataFrame containing evaluation metrics for all models.
    """
    # Define models to train
    models_to_train = {
        "Logistic Regression (Fraud)": ("logistic_regression", X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud),
        "Logistic Regression (Credit)": ("logistic_regression", X_train_credit, y_train_credit, X_test_credit, y_test_credit),
        "Decision Tree (Fraud)": ("decision_tree", X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud),
        "Decision Tree (Credit)": ("decision_tree", X_train_credit, y_train_credit, X_test_credit, y_test_credit),
        "Random Forest (Fraud)": ("random_forest", X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud),
        "Random Forest (Credit)": ("random_forest", X_train_credit, y_train_credit, X_test_credit, y_test_credit),
        "XGBoost (Fraud)": ("xgboost", X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud),
        "XGBoost (Credit)": ("xgboost", X_train_credit, y_train_credit, X_test_credit, y_test_credit),
        "MLP (Fraud)": ("mlp", X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud),
        "MLP (Credit)": ("mlp", X_train_credit, y_train_credit, X_test_credit, y_test_credit),
    }
    
    # Initialize a dictionary to store metrics
    metrics_dict = {}
    
    # Train and evaluate each model
    for name, (model_type, X_train, y_train, X_test, y_test) in models_to_train.items():
        print(f"Training and evaluating {name}...")
        model = train_model(model_type, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics_dict[name] = metrics
    
    # Convert metrics to DataFrame for easier viewing
    results_df = pd.DataFrame(metrics_dict).T  # Transpose for better format
    return results_df


def save_model(model, model_path):
    """Save the trained model to a specified path."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

