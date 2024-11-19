import seaborn as sns
import matplotlib.pyplot as plt
def display_skewness(data):
    skew_kurt_summary = {}
    for col in data.select_dtypes(include=['number']).columns:
        skew_kurt_summary[col] = {
            "Skewness": data[col].skew(),
            "Kurtosis": data[col].kurt()
        }
        print(f"{col}\nSkewness: {data[col].skew()}\nKurtosis: {data[col].kurt()}")
    return skew_kurt_summary

