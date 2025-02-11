
    # Import libraries
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def load_data(data_path, model_path):
    """
    Load the dataset and the trained model.
    
    Parameters:
    - data_path: Path to the dataset CSV file.
    - model_path: Path to the trained model file.
    
    Returns:
    - fraud_data: Loaded dataset as a DataFrame.
    - model: Loaded trained model.
    """
    fraud_data = pd.read_csv(data_path)
    model = joblib.load(model_path)
    return fraud_data, model

def preprocess_data(fraud_data):
    """
    Preprocess the dataset by dropping irrelevant columns and encoding categorical variables.
    
    Parameters:
    - fraud_data: Raw dataset as a DataFrame.
    
    Returns:
    - X: Features DataFrame.
    - y: Target Series.
    """
    # Drop non-numeric or irrelevant columns
    fraud_data_encoded = fraud_data.drop(columns=['signup_time', 'purchase_time', 'device_id', 'user_id'])
    
    # Convert categorical variables to one-hot encoding
    fraud_data_encoded = pd.get_dummies(fraud_data_encoded, columns=['source', 'browser', 'sex'], drop_first=True)
    
    # Separate features and target
    X = fraud_data_encoded.drop(columns=['class'])
    y = fraud_data_encoded['class']
    
    return X, y

def shap_explainability(model, X, sample_size=100, random_state=42):
    """
    Perform SHAP explainability on the model using a sample of the data.
    
    Parameters:
    - model: Trained model.
    - X: Features DataFrame.
    - sample_size: Number of samples to use for SHAP explainability.
    - random_state: Random seed for reproducibility.
    
    Returns:
    - explainer: SHAP TreeExplainer.
    - shap_values_fraud: SHAP values for the fraud class.
    - X_sample: Sampled features DataFrame.
    """
    # Sample the data
    X_sample = X.sample(sample_size, random_state=random_state)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Verify the structure of shap_values
    if isinstance(shap_values, list):
        print("SHAP values is a list. Using shap_values[1] for the fraud class.")
        shap_values_fraud = shap_values[1]
    else:
        print("SHAP values is a single array. Extracting SHAP values for class 1 (fraud).")
        shap_values_fraud = shap_values[:, :, 1]  # Extract SHAP values for class 1
    
    # Verify shapes
    print("Shape of shap_values_fraud:", shap_values_fraud.shape)
    print("Shape of X_sample:", X_sample.shape)
    
    # Ensure shapes match
    assert shap_values_fraud.shape == X_sample.shape, "Shapes of shap_values and X_sample do not match!"
    
    return explainer, shap_values_fraud, X_sample

def plot_shap_summary(shap_values_fraud, X_sample):
    """
    Plot SHAP summary plot.
    
    Parameters:
    - shap_values_fraud: SHAP values for the fraud class.
    - X_sample: Sampled features DataFrame.
    """
    print("SHAP Summary Plot")
    shap.summary_plot(shap_values_fraud, X_sample, plot_type="bar")

def plot_shap_force(explainer, shap_values_fraud, X_sample, index=5):
    """
    Plot SHAP force plot for a specific instance.
    
    Parameters:
    - explainer: SHAP TreeExplainer.
    - shap_values_fraud: SHAP values for the fraud class.
    - X_sample: Sampled features DataFrame.
    - index: Index of the instance to explain.
    """
    print("SHAP Force Plot")
    shap.initjs()
    shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values_fraud[index],
        features=X_sample.iloc[index],
        matplotlib=True
    )

def plot_shap_dependence(shap_values_fraud, X_sample, feature="purchase_value"):
    """
    Plot SHAP dependence plot for a specific feature.
    
    Parameters:
    - shap_values_fraud: SHAP values for the fraud class.
    - X_sample: Sampled features DataFrame.
    - feature: Feature to plot dependence for.
    """
    print("SHAP Dependence Plot")
    shap.dependence_plot(feature, shap_values_fraud, X_sample)

def lime_explainability(model, X, instance_index=5, num_features=10):
    """
    Perform LIME explainability on the model for a specific instance.
    
    Parameters:
    - model: Trained model.
    - X: Features DataFrame.
    - instance_index: Index of the instance to explain.
    - num_features: Number of features to show in the explanation.
    
    Returns:
    - lime_exp: LIME explanation object.
    """
    # Initialize LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, 
        feature_names=X.columns,
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        X.iloc[instance_index].values, 
        model.predict_proba, 
        num_features=num_features
    )
    
    return lime_exp

def plot_lime_explanation(lime_exp):
    """
    Display LIME explanation for a specific instance.
    
    Parameters:
    - lime_exp: LIME explanation object.
    """
    print("Display LIME explanation...")
    lime_exp.show_in_notebook(show_table=True)