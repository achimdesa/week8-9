import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_fraud_data(file_path):
    """Load the fraud data from a CSV file."""
    fraud_data = pd.read_csv(file_path)
    print("Fraud Data Sample:")
    display = print(fraud_data.head(), "\n")
    return fraud_data,display


def prepare_data(fraud_data):
    """Prepare the data by encoding categorical variables and dropping unnecessary columns."""
    fraud_data_encoded = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)
    fraud_data_encoded = fraud_data_encoded.drop(columns=['signup_time', 'purchase_time', 'device_id'])
    X_fraud = fraud_data_encoded.drop(columns=['class'])  # Features
    y_fraud = fraud_data_encoded['class']  # Target
    return X_fraud, y_fraud

def split_data(X_fraud, y_fraud):
    """Split the dataset into training and testing sets."""
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42
    )
    return X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud

def train_random_forest(X_train_fraud, y_train_fraud):
    """Train a Random Forest model."""
    print("Training Random Forest Model on Fraud Data...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_fraud, y_train_fraud)
    return rf_model

def evaluate_model(rf_model, X_test_fraud, y_test_fraud):
    """Make predictions and evaluate the model."""
    y_pred_fraud = rf_model.predict(X_test_fraud)
    print("Random Forest - Fraud Data - Classification Report:")
    print(classification_report(y_test_fraud, y_pred_fraud))
    print(f"Random Forest Accuracy on Fraud Data: {accuracy_score(y_test_fraud, y_pred_fraud)}\n")
    return y_pred_fraud

def plot_feature_importance(rf_model, X_train_fraud):
    """Plot the feature importance from the Random Forest model."""
    feature_importances = rf_model.feature_importances_
    features = X_train_fraud.columns
    importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances_df)
    plt.title('Feature Importances in Random Forest (Fraud Data)')
    plt.show()

def save_model(rf_model, model_path):
    """Save the trained model to a specified path."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(rf_model, model_path)
    print(f"Random Forest model saved at: {model_path}")