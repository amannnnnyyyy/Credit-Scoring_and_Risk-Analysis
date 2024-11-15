import matplotlib.pyplot as plt
import seaborn as sns


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



def plot_fraud_result(data):
    # Histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data['FraudResult'], bins=10, kde=True)
    plt.title('Histogram of Values')

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data['FraudResult'])
    plt.title('Box Plot of Values')

def pricing_strategy(data):
    # Histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data['PricingStrategy'], bins=10, kde=True)
    plt.title('Histogram of PricingStrategy')

    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data['PricingStrategy'])
    plt.title('Box Plot of PricingStrategy')

    plt.show()

    plt.show()


def plot_product_category(data):
    plt.figure(figsize=(12, 8))
    # Histogram
    plt.subplot(2, 1, 1)
    sns.histplot(data['ProductCategory'], bins=10, kde=True)
    plt.title('Histogram of ProductCategory')

    # Box Plot
    plt.subplot(2, 1, 2)
    sns.boxplot(x=data['ProductCategory'])
    plt.title('Box Plot of ProductCategory')

    plt.tight_layout()
    plt.show()


def correlation_analysis(dataframe):
    numerical_cols = dataframe.select_dtypes(include=['number']).columns
    dataframe = dataframe[numerical_cols[1:]]
    correlation_matrix = dataframe.corr()

    # Set plot style
    sns.set(style='whitegrid')

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()