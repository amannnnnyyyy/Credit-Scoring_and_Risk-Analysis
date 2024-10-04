# Credit Scoring Model

## Overview

This project aims to develop a credit scoring model to assess the creditworthiness of individuals based on their transaction history and other relevant features. The model employs various machine learning techniques to predict the likelihood of default.

## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Data](#data)
- [Installation](#installation)

## Features

- **Data Preprocessing**: Handling missing values, feature engineering, and scaling.
- **Feature Engineering**: Creation of aggregate features and time-based features.
- **Encoding**: One-hot and label encoding of categorical variables.
- **Modeling**: Implementation of various machine learning algorithms to predict credit scores.

## Technologies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Data

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)[Not available now]. It contains features such as transaction amounts, customer information, and transaction timestamps.

### Dataset Features
- `TransactionStartTime`: Timestamp of the transaction.
- `CustomerId`: Unique identifier for each customer.
- `Amount`: Transaction amount.
- `ProductCategory`: Category of the product purchased.
- `ChannelId`: Channel through which the transaction was made.
- `CurrencyCode`: Currency in which the transaction was made.
- And others

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/amannnnnyyyy/Credit-Scoring_and_Risk-Analysis.git
   cd Credit-Scoring_and_Risk-Analysis

2. pip install -r requirements.txt
