import numpy as np

def calculate_recency(df, customer_id_col, transaction_time_col):
    current_date = df[transaction_time_col].max()  # Latest date in the dataset
    recency = df.groupby(customer_id_col)[transaction_time_col].apply(lambda x: (current_date - x.max()).days)
    return recency

def calculate_frequency(df, customer_id_col, transaction_id_col):
    frequency = df.groupby(customer_id_col)[transaction_id_col].nunique()
    return frequency

def calculate_monetary(df, customer_id_col, transaction_amount_col):
    monetary = df.groupby(customer_id_col)[transaction_amount_col].sum()
    return monetary

def calculate_seasonality(df, customer_id_col, transaction_time_col):
    # Count the number of transactions per month for each customer
    df['Month'] = df[transaction_time_col].dt.to_period('M')  # Create a 'Month' column
    seasonality = df.groupby([customer_id_col, 'Month']).size().reset_index(name='Transactions')
    # Count the number of months with transactions to get a seasonality score
    seasonality_score = seasonality.groupby(customer_id_col)['Transactions'].count()
    return seasonality_score


def calculate_risk_category(data):
    median_recency = data['Recency'].median()
    median_frequency = data['Frequency'].median()
    median_monetary = data['Monetary'].median()
    median_seniority = data['Seasonality'].median()
    data['RiskCategory'] = (
            (data['Frequency'] >= median_frequency) &
            (data['Monetary'] >= median_monetary) &
            (data['Recency'] >= median_recency) &
            (data['Seasonality'] >= median_seniority)
    ).map({True: 'Good', False: 'Bad'})

    # Display the updated DataFrame with Risk Categories
    print(data[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Seasonality', 'RiskCategory']].head())


def calculate_woe(data):
    woe_df = data.groupby('TotalRFMS').agg(
        Total=('RiskCategory', 'size'),
        Bad=('RiskCategory', lambda x: (x == 'Bad').sum())
    ).reset_index()

    # Calculate Good counts
    woe_df['Good'] = woe_df['Total'] - woe_df['Bad']

    # Print the intermediate results
    print(woe_df)
    return woe_df


def calculate_woe_option(data, feature, target):
    # Group by the specified feature and calculate totals
    woe_df = data.groupby(feature).agg(
        Total=('RiskCategory', 'size'),
        Bad=('RiskCategory', lambda x: (x == 'Bad').sum())
    ).reset_index()

    # Calculate Good counts
    woe_df['Good'] = woe_df['Total'] - woe_df['Bad']

    # Total Bad and Good with Laplace smoothing
    total_bad = woe_df['Bad'].sum() + 1  # Add 1 for Laplace smoothing
    total_good = woe_df['Good'].sum() + 1

    # Calculate WoE with handling for zero Good or Bad
    woe_df['WoE'] = np.where(
        (woe_df['Good'] == 0) & (woe_df['Bad'] == 0),
        0,  # Both are zero, set WoE to 0
        np.where(
            woe_df['Bad'] == 0,
            np.log(total_good / (total_bad + 1)),  # Handle case where Bad is zero
            np.where(
                woe_df['Good'] == 0,
                np.log((total_good + 1) / total_bad),  # Handle case where Good is zero
                np.log((woe_df['Good'] / total_good) / (woe_df['Bad'] / total_bad))  # Normal case
            )
        )
    )

    return woe_df[[feature, 'WoE']]

def calculate_total_RFMS(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Calculate Recency (days since last transaction)
    recency_df = data.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency_df['Recency'] = (data['TransactionStartTime'].max() - recency_df['TransactionStartTime']).dt.days

    # Calculate Frequency (number of transactions)
    frequency_df = data.groupby('CustomerId')['TransactionId'].count().reset_index()
    frequency_df.columns = ['CustomerId', 'Frequency']

    # Calculate Monetary Value (total transaction amount)
    monetary_df = data.groupby('CustomerId')['Amount'].sum().reset_index()
    monetary_df.columns = ['CustomerId', 'Monetary']

    # Merge RFMS data
    rfms_df = recency_df.merge(frequency_df, on='CustomerId').merge(monetary_df, on='CustomerId')

    # Normalize RFMS scores
    rfms_df['RecencyScore'] = pd.qcut(rfms_df['Recency'], 4, labels=[4, 3, 2, 1])  # Lower recency is better
    rfms_df['FrequencyScore'] = pd.qcut(rfms_df['Frequency'], 4, labels=[1, 2, 3, 4])  # Higher frequency is better
    rfms_df['MonetaryScore'] = pd.qcut(rfms_df['Monetary'], 4, labels=[1, 2, 3, 4])  # Higher monetary is better

    # Calculate Total RFMS Score
    rfms_df['TotalRFMS'] = rfms_df['RecencyScore'].astype(int) + rfms_df['FrequencyScore'].astype(int) + rfms_df['MonetaryScore'].astype(int)

    # Display RFMS DataFrame
    print(rfms_df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'TotalRFMS']].head())
    return rfms_df