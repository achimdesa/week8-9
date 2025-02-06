import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def load_datasets():
    """Load the datasets from specified paths."""
    try:
        
        creditcard_data = pd.read_csv('../data/creditcard.csv')
        fraud_data = pd.read_csv('../data/Fraud_Data.csv')
        ip_data = pd.read_csv('../data/IpAddress_to_Country.csv')
        print(f"Data loaded successfully. Credit card Shape: {creditcard_data.shape}, Fraud data Shape: {fraud_data.shape}, IP data Shape: {ip_data.shape}")
        return creditcard_data, fraud_data, ip_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def display_samples(creditcard_data, fraud_data, ip_data):
    """Display the first few rows of each dataset."""
    print("Credit Card Data Sample:")
    print(creditcard_data.head(), "\n")
    print("Fraud Data Sample:")
    print(fraud_data.head(), "\n")
    print("IP Address Data Sample:")
    print(ip_data.head(), "\n")

def check_missing_values(creditcard_data, fraud_data, ip_data):
    """Check for missing values in the datasets."""
    print("Credit Card Data Missing Values:\n", creditcard_data.isnull().sum(), "\n")
    print("Fraud Data Missing Values:\n", fraud_data.isnull().sum(), "\n")
    print("IP Address Data Missing Values:\n", ip_data.isnull().sum(), "\n")

def clean_data(creditcard_data, fraud_data, ip_data):
    """Drop missing values and duplicates from datasets."""
    creditcard_data.dropna(inplace=True)
    fraud_data.dropna(inplace=True)
    ip_data.dropna(inplace=True)

    creditcard_data.drop_duplicates(inplace=True)
    fraud_data.drop_duplicates(inplace=True)
    ip_data.drop_duplicates(inplace=True)

    # Convert relevant columns to datetime
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

    return creditcard_data, fraud_data, ip_data

def plot_purchase_value_distribution(fraud_data):
    """Plot the distribution of purchase values."""
    print("Plotting Distribution of Purchase Value...")
    fraud_data['purchase_value'].hist(bins=50)
    plt.title('Distribution of Purchase Value')
    plt.xlabel('Purchase Value')
    plt.ylabel('Frequency')
    plt.show()

def plot_purchase_value_vs_class(fraud_data):
    """Plot purchase value against fraud class."""
    print("Plotting Purchase Value vs Fraud Class...")
    sns.boxplot(x='class', y='purchase_value', data=fraud_data)
    plt.title('Purchase Value vs Fraud Class')
    plt.xlabel('Fraud Class')
    plt.ylabel('Purchase Value')
    plt.show()


def convert_ip_to_integer(fraud_data):
    """Convert IP addresses to integer format, handling NaNs."""
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(str).apply(
        lambda ip: int(ip.replace('.', '')) if ip != 'nan' else 0
    )
    return fraud_data

def merge_ip_data(fraud_data, ip_data):
    """Merge fraud data with IP Address dataset."""
    merged_data = pd.merge(fraud_data, ip_data, how='left',
                            left_on='ip_address', right_on='lower_bound_ip_address')
    # Drop unnecessary columns from merge
    merged_data.drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis=1, inplace=True)
    return merged_data

def feature_engineering(fraud_data):
    """Create new features: transaction frequency, velocity, time-based features."""
    fraud_data['transaction_frequency'] = fraud_data.groupby('user_id')['user_id'].transform('count')
    fraud_data['transaction_velocity'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    return fraud_data

def normalize_features(fraud_data):
    """Normalize purchase_value and transaction_velocity using MinMaxScaler."""
    scaler = MinMaxScaler()
    fraud_data[['purchase_value', 'transaction_velocity']] = scaler.fit_transform(
        fraud_data[['purchase_value', 'transaction_velocity']]
    )
    return fraud_data

def one_hot_encode(fraud_data):
    """One-hot encode categorical columns."""
    fraud_data = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)
    return fraud_data

def display_final_data(fraud_data):
    """Display the final fraud data with new features."""
    print("Fraud Data with New Features:")
    print(fraud_data.head(), "\n")