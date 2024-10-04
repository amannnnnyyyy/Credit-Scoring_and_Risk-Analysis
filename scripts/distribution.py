import seaborn as sns
import matplotlib.pyplot as plt
def visualize_numerical_features(dataframe):
    numerical_cols = dataframe.select_dtypes(include=['number']).columns
    
    # Set plot style
    sns.set(style='whitegrid')
    
    # Create a figure to hold the subplots for histogram and KDE
    plt.figure(figsize=(15, 4 * len(numerical_cols)))
    
    # Histograms with KDE
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(len(numerical_cols), 1, i)
        sns.histplot(dataframe[col], bins=10, kde=True)
        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

    plt.tight_layout()
    plt.show()
    
    # Create a figure to hold the box plots
    plt.figure(figsize=(15, 5))
    
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(1, len(numerical_cols), i)
        sns.boxplot(x=dataframe[col])
        plt.title(f'Box Plot of {col}', fontsize=16)

    plt.tight_layout()
    plt.show()


def visualize_categorical_features(dataframe):
    # Check for empty DataFrame
    if dataframe.empty:
        print("The DataFrame is empty.")
        return
    
    categorical_cols = dataframe.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("No categorical columns found.")
        return

    sns.set(style='whitegrid')

    # Create a figure to hold the subplots for count plots
    plt.figure(figsize=(15, 4 * len(categorical_cols)))

    # Count plots for categorical features
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(len(categorical_cols), 1, i)
        sns.countplot(data=dataframe, x=col)
        plt.title(f'Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Count', fontsize=14)

    plt.tight_layout()
    plt.show()

