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