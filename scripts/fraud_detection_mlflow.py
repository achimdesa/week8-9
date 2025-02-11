import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Function to preprocess Fraud_Data
def preprocess_fraud_data(fraud_data):
    fraud_data_encoded = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)
    fraud_data_encoded = fraud_data_encoded.drop(columns=['signup_time', 'purchase_time', 'device_id'])
    X = fraud_data_encoded.drop(columns=['class'])
    y = fraud_data_encoded['class']
    return X, y

# Function to preprocess Credit_Data
def preprocess_credit_data(credit_data):
    X = credit_data.drop(columns=['Class'])
    y = credit_data['Class']
    return X, y

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name, model_name):
    # Train the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log parameters and metrics
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("model", model_name)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("auc_roc", roc_auc_score(y_test, y_pred))

# Function to run nested experiments for a dataset
def run_experiments(X_train, X_test, y_train, y_test, dataset_name):
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier()
    }

    # Parent run for the dataset
    with mlflow.start_run(run_name=f"{dataset_name} Experiment"):
        for model_name, model in models.items():
            # Nested run for each model
            with mlflow.start_run(run_name=f"{model_name} - {dataset_name}", nested=True):
                train_and_evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name, model_name)

# Main function
def main():
    # Load datasets
    fraud_data = pd.read_csv("../data/Fraud_Data.csv")
    credit_data = pd.read_csv("../data/creditcard.csv")

    # Preprocess Fraud_Data
    X_fraud, y_fraud = preprocess_fraud_data(fraud_data)
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

    # Preprocess Credit_Data
    X_credit, y_credit = preprocess_credit_data(credit_data)
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

    # Run experiments for Fraud_Data
    run_experiments(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud, "Fraud_Data")

    # Run experiments for Credit_Data
    run_experiments(X_train_credit, X_test_credit, y_train_credit, y_test_credit, "Credit_Data")

# Entry point
if __name__ == "__main__":
    main()