# week8-9
10 Academy Kifiya AI mastery training program week 8&amp;9 challenge


## Project Overview
This project is part of the 10 Academy AI Mastery program's Week 8 & 9 challenge, focused on improving fraud detection in e-commerce and banking transactions. The goal is to create accurate machine learning models that can detect fraudulent transactions based on transaction data.

This repository contains the code and results for:
- **Task 1: Data Analysis and Preprocessing**
- **Task 2: Model Building and Training**
- **Task 3: Model Explainability**
- **Task 4: Model Deployment and API Development**
- **Task 5: Build a Dashboard with Flask and Dash**

## Datasets Used
1. **Credit Card Data** (`creditcard.csv`): Contains anonymized bank transaction features and fraud labels (`Class`).
2. **Fraud Data** (`Fraud_Data.csv`): Includes e-commerce transaction data with user details, transaction timestamps, purchase values, and fraud labels (`class`).
3. **IP Address Data** (`IpAddress_to_Country.csv`): Maps IP address ranges to country names.

## Steps Completed in Task 1
1. **Data Loading**: 
   The datasets were loaded into pandas dataframes for analysis and manipulation.
   - Credit Card Data: Contains anonymized features (V1 to V28), transaction amounts, and fraud labels.
   - Fraud Data: Contains user, device, and transaction details, as well as fraud indicators.
   - IP Address Data: Contains IP address range mappings to countries.

2. **Data Cleaning**: 
   The datasets were checked for missing values and duplicates, and any duplicates were removed. No missing values were found in any of the datasets.

3. **Exploratory Data Analysis (EDA)**:
   - **Univariate Analysis**: A histogram was used to examine the distribution of `purchase_value`.
   - **Bivariate Analysis**: A box plot was created to explore the relationship between `purchase_value` and fraud (`class`).

4. **Geolocation Merging**: 
   The `Fraud_Data` was merged with the `IpAddress_to_Country.csv` dataset to provide geographic context for the transactions, based on the IP addresses.

5. **Feature Engineering**:
   - **Transaction Frequency**: The number of transactions per user.
   - **Transaction Velocity**: The time difference between the signup and purchase times.
   - **Time-Based Features**: Hour of day and day of week were extracted from the purchase timestamp.

6. **Data Normalization and Encoding**:
   - Numeric features (`purchase_value`, `transaction_velocity`) were normalized using Min-Max scaling.
   - Categorical features (`source`, `browser`, `sex`) were one-hot encoded for use in machine learning models.


## Steps Completed in Task 2
1. **Model Selection and Training**:
   Several machine learning models were trained to detect fraudulent transactions, using both the **Credit Card Data** and **Fraud Data**.
   
   - **Random Forest**: A Random Forest model was trained on the **Fraud Data** to predict fraud based on the engineered features.
   - **Logistic Regression and Gradient Boosting**: These models were trained on the **Credit Card Data** to predict fraud based on the PCA-transformed features.

2. **Model Evaluation**:
   The models were evaluated using classification metrics like **accuracy**, **precision**, **recall**, and **F1-score** to assess their performance in identifying fraudulent transactions.

   - **Random Forest** performed well on the **Fraud Data** with high accuracy in detecting fraudulent transactions.
   - **Logistic Regression and Gradient Boosting** were evaluated on the **Credit Card Data**.

3. **Saving the Model**:
   The trained **Random Forest** model was saved to the `../models` directory for future use (e.g., deploying it in an API or batch prediction).

## Running the Project

### Prerequisites
You will need the following Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib

## How to Run the Code

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/SolomonZinabu/week-8-9
    ```

2. **Navigate** to the project directory:

    ```bash
    cd week-8-9
    ```

3. Ensure you have the necessary datasets (`creditcard.csv`, `Fraud_Data.csv`, and `IpAddress_to_Country.csv`) in the `/data` folder relative to the task notebook.

4. **To execute Task 1** (Data Analysis and Preprocessing):

    ```bash
    jupyter notebook task-1.ipynb
    ```

5. **To execute Task 2** (Model Building and Training), including saving the Random Forest model:

    ```bash
    jupyter notebook task2.ipynb
    ```

## Results

### Task 1 Results

- **Data Cleaning**: All datasets were free from missing values, and duplicates were removed.
- **EDA**: Various visualizations were produced to understand the relationship between features like `purchase_value` and fraud (`class`).
- **Feature Engineering**: New features such as transaction frequency, velocity, and time-based features were added to the dataset to enhance model training.
- **Geolocation**: Transactions were enriched with country information based on IP address ranges.

### Task 2 Results

- **Random Forest**: A Random Forest model was trained on the **Fraud Data** to identify fraudulent transactions, achieving strong performance with high accuracy.
- **Logistic Regression and Gradient Boosting**: These models were trained on the **Credit Card Data** and evaluated using classification metrics.
- **Model Saving**: The Random Forest model was saved for future tasks in `../models/random_forest_fraud_model.pkl`.

## Next Steps

With the trained models saved, the next step is to use these models for real-time fraud detection in future tasks. This may involve serving predictions via an API or further tuning the models for improved accuracy.

## Task 3: Model Explainability

### Objective
Enhance the interpretability of the trained fraud detection model using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to provide insights into feature importance and model decision-making.

### Steps Completed

1. **SHAP Explainability**:
   - **Summary Plot**: Generated a bar plot to display the overall feature importance, providing a high-level view of which features most impact the model’s fraud predictions.
   - **Force Plot**: Visualized how individual feature values contribute to specific predictions, using SHAP force plots to illustrate the feature impact on single-instance fraud predictions.
   - **Dependence Plot**: Created SHAP dependence plots to understand relationships between certain features (like `purchase_value`) and the model’s fraud probability predictions.

2. **LIME Explainability**:
   - Initialized a **LIME Tabular Explainer** to examine the model’s predictions on individual cases, helping to interpret which features influenced each specific prediction.
   - Generated LIME explanations for selected instances, revealing feature contributions in an interpretable, case-by-case format.

### Results
- **Key Insights**: The analysis identified `purchase_value` and `transaction_frequency` as significant indicators in the fraud detection model, with a strong correlation to fraud likelihood.
- **Enhanced Interpretability**: By applying SHAP and LIME, we gained clear insights into the model’s decision process, enabling transparency and trust in its predictions.

These interpretability tools empowered us to better understand the feature interactions and contributions within our fraud detection model, preparing it for deployment and real-world application in subsequent tasks.

## Task 4: Model Deployment and API Development

### Objective
Deploy the trained fraud detection model as a Flask API, making it accessible for real-time fraud detection requests. Containerize the API using Docker for portability and scalability.

### Steps Completed

1. **Flask API Development**:
   - Created a Flask API with endpoints to serve model predictions and check API status.
   - Integrated logging to track requests, making it easy to monitor predictions and identify any issues.

2. **Dockerization**:
   - Developed a `Dockerfile` to containerize the Flask API.
   - The Docker container provides a portable, scalable solution that allows the API to be deployed in various environments.

### How to Run

1. Build the Docker image:
   ```bash
   docker build -t fraud-detection-api .


ask 5: Dashboard Creation
Objective
Develop an interactive dashboard using Dash to visualize fraud detection insights, enabling stakeholders to monitor fraud trends and analyze key metrics in real-time.

Steps Completed
Dashboard Development:

Built a Dash dashboard integrated with a Flask backend to visualize key fraud insights interactively.
The dashboard provides an accessible, web-based interface for stakeholders to explore trends and metrics.
Dashboard Features:

Time Series Visualization: Displays fraud cases over time to monitor temporal trends.
Geographical Analysis: Maps fraud cases by country for geographic insights.
Device/Browser Insights: Shows fraud distribution across different devices and browsers.