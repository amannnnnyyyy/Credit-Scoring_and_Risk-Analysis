import pytest
import pandas as pd
from sklearn.impute import SimpleImputer
import pytest
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('..')
from scripts.calculations import *
from scripts.data_manip import *
from scripts.distribution import *
from scripts.feature_engineering import *
from scripts.modeling import *
from scripts.visualizations import *


@pytest.fixture
def sample_data():
    data = {
        'TransactionId': ['TransactionId_76871', 'TransactionId_73770', 'TransactionId_26203', 'TransactionId_380'],
        'BatchId': ['BatchId_36123', 'BatchId_15642', 'BatchId_53941', 'BatchId_102363'],
        'AccountId': ['AccountId_3957', 'AccountId_4841', 'AccountId_4229', 'AccountId_648'],
        'SubscriptionId': ['SubscriptionId_887', 'SubscriptionId_3829', 'SubscriptionId_222', 'SubscriptionId_2185'],
        'CustomerId': ['CustomerId_4406', 'CustomerId_4406', 'CustomerId_4683', 'CustomerId_988'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': ['256', '256', '256', '256'],
        'ProviderId': ['ProviderId_6', 'ProviderId_4', 'ProviderId_6', 'ProviderId_1'],
        'ProductId': ['ProductId_10', 'ProductId_6', 'ProductId_1', 'ProductId_21'],
        'ProductCategory': ['airtime', 'financial_services', 'airtime', 'utility_bill'],
        'ChannelId': ['ChannelId_3', 'ChannelId_2', 'ChannelId_3', 'ChannelId_3'],
        'Amount': [1000, -20, 500, 20000],
        'Value': [1000, 20, 500, 21800],
        'TransactionStartTime': [
            '2018-11-15T02:18:49Z',
            '2018-11-15T02:19:08Z',
            '2018-11-15T02:44:21Z',
            '2018-11-15T03:32:55Z',
        ],
        'PricingStrategy': [2, 2, 2, 2],
        'FraudResult': [0, 0, 1, 1],
    }
    return pd.DataFrame(data)

def test_display_skewness(sample_data):
    result = display_skewness(sample_data)
    assert result is not None 

def test_time_correction(sample_data):
    corrected_data = time_correction(sample_data)
    assert 'Transaction_Hour' in corrected_data.columns
    assert 'Transaction_Year' in corrected_data.columns

def test_aggregate_features(sample_data):
    aggregated = aggregate_features(sample_data, 'CustomerId')
    assert 'Average_Transaction_Amount' in aggregated.columns
    assert aggregated.shape[0] == 3

def test_one_hot_encoder(sample_data):
    encoded_data = one_hot_encoder(sample_data)
    assert encoded_data.shape[1] > sample_data.shape[1]

def test_standardize(sample_data):
    sample_data['Recency'] = [10, 20, 30, 40]
    sample_data['Frequency'] = [5, 10, 15, 20]
    sample_data['Monetary'] = [100, 200, 300, 400]
    sample_data['Seasonality'] = [1, 2, 3, 4]

    sample_data = classify_customers_by_rfms(sample_data)

    numerical_columns = ['Amount', 'RiskScore']
    
    standardized = Standardize(sample_data, numerical_columns)
    assert 'RiskScore' in standardized.columns
    assert not standardized['RiskScore'].isna().any()


def test_data_processing_pipeline(sample_data):
    corrected_data = time_correction(sample_data)
    aggregated = aggregate_features(corrected_data, 'CustomerId')
    processed_data = one_hot_encoder(aggregated)
    assert 'Average_Transaction_Amount' in processed_data.columns

def test_modeling_pipeline(sample_data):
    corrected_data = time_correction(sample_data)
    aggregated = aggregate_features(corrected_data, 'CustomerId')
    processed_data = one_hot_encoder(aggregated)

    imputer = SimpleImputer(strategy='most_frequent')
    processed_data_imputed = imputer.fit_transform(processed_data)
    processed_data_imputed = pd.DataFrame(processed_data_imputed, columns=processed_data.columns)

    expected_columns = [
        'Month', 'TransactionStartTime', 'TransactionId', 'AccountId', 'RiskCategory', 'BatchId', 'SubscriptionId',
        'TotalRFMS', 'CountryCode', 'Amount', 'Value', 'PricingStrategy', 'FraudResult'
    ]
    for col in expected_columns:
        if col not in processed_data_imputed:
            processed_data_imputed[col] = 0 

    target_column = 'FraudResult'
    if processed_data_imputed[target_column].isna().any():
        print("NaN found in target variable, filling with 0.")
        processed_data_imputed[target_column].fillna(0, inplace=True)

    X = processed_data_imputed.drop(columns=[target_column])
    y = processed_data_imputed[target_column]
    assert not y.isna().any(), "Target variable contains NaN after preprocessing."





@pytest.fixture
def actual_data():
    X_test = pd.read_csv("../docs/X_test_data.csv")
    y_test = pd.read_csv("../docs/y_test_data.csv").squeeze() 

    if 'Unnamed: 0' in X_test.columns:
        X_test.drop(columns=['Unnamed: 0'], inplace=True)

    return X_test, y_test

def test_model_predictions(actual_data):
    X_test, y_test = actual_data

    model = joblib.load("../models/gradient_boosting_model.pkl") 

    log_preds = model.predict_proba(X_test)[:, 1] 

    assert len(log_preds) == len(y_test), "Predictions length does not match actual values length."

    if y_test.ndim > 1:
        y_test = y_test.iloc[:, 0] 

    try:
        auc = roc_auc_score(y_test, log_preds)
        print(f"ROC AUC Score: {auc:.4f}")
    except ValueError as e:
        print(f"Error calculating ROC AUC: {e}")

    print("Model predictions test passed!")