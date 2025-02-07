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
