import pandas as pd

def find_missing_values(data):
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_summary = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})

    print(f"Missing Values Summary: {missing_values}")
    print(missing_summary[missing_summary['Missing Values'] > 0])