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