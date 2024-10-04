import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def aggregate_features(data,by):
    aggregate_features = data.groupby(by).agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('TransactionId', 'count'),
        Std_Deviation_Transaction_Amount=('Amount', 'std')
    ).reset_index()

    return aggregate_features

def time_correction(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Extract features from the TransactionStartTime
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year


def one_hot_encoder(data):
    encoder = OneHotEncoder(drop='first', dtype=int)

    # Select categorical columns for encoding
    categorical_columns = ['ProductCategory', 'ChannelId', 'CurrencyCode']

    # Fit and transform the data
    encoded_features = encoder.fit_transform(data[categorical_columns])

    # Check the shape of encoded_features
    print(f"Shape of encoded features: {encoded_features.shape}")

    # Create a DataFrame from the encoded features
    # Ensure to use get_feature_names_out() correctly
    encoded_df = pd.DataFrame(encoded_features.toarray(), 
                            columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the original DataFrame with the new encoded DataFrame
    df_encoded = pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Drop original categorical columns if no longer needed
    df_encoded.drop(columns=categorical_columns, inplace=True)

    return df_encoded


def label_encoder(data):
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()

    # Encode categorical variables
    data['ProductCategory_Label'] = label_encoder.fit_transform(data['ProductCategory'])
    data['ChannelId_Label'] = label_encoder.fit_transform(data['ChannelId'])
    data['CurrencyCode_Label'] = label_encoder.fit_transform(data['CurrencyCode'])
    return data




def null_value_imputing_KNN(data):

    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=5)

    # Impute missing values
    df_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=[np.number])), 
                            columns=data.select_dtypes(include=[np.number]).columns)
    return df_imputed



def scaling(data):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    # Initialize MinMaxScaler
    min_max_scaler = MinMaxScaler()

    # Apply normalization
    df_normalized = data.copy()  # Create a copy to avoid modifying the original DataFrame
    df_normalized[numerical_columns] = min_max_scaler.fit_transform(data[numerical_columns])

    return [df_normalized,numerical_columns]



def Standardize(data,numerical_columns):
    # Initialize StandardScaler
    standard_scaler = StandardScaler()

    # Apply standardization
    df_standardized = data.copy()  # Create a copy to avoid modifying the original DataFrame
    df_standardized[numerical_columns] = standard_scaler.fit_transform(data[numerical_columns])
    return df_standardized