import seaborn as sns
import matplotlib.pyplot as plt
def display_skewness(data):
    for col in data.select_dtypes(include=['number']).columns:
        skewness = data[col].skew()
        kurtosis = data[col].kurtosis()
        print(col)
        print(f'Skewness: {skewness}')
        print(f'Kurtosis: {kurtosis}')

