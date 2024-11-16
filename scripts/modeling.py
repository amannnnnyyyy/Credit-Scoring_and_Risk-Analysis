from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib


def prepare_for_model(data):
    X = data.drop(
        ['Month', 'TransactionStartTime', 'TransactionId', 'AccountId', 'CustomerId', 'RiskCategory', 'BatchId',
         'SubscriptionId'], axis=1)  # Exclude identifiers and target
    X = data[['TotalRFMS', 'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3',
              'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5', 'ProviderId_ProviderId_6',
              'ProductId_ProductId_10', 'ProductId_ProductId_11', 'ProductId_ProductId_12',
              'ProductId_ProductId_13', 'ProductId_ProductId_14', 'ProductId_ProductId_15',
              'ProductId_ProductId_16', 'ProductId_ProductId_19', 'ProductId_ProductId_2',
              'ProductId_ProductId_20', 'ProductId_ProductId_21', 'ProductId_ProductId_22',
              'ProductId_ProductId_23', 'ProductId_ProductId_24', 'ProductId_ProductId_27',
              'ProductId_ProductId_3', 'ProductId_ProductId_4', 'ProductId_ProductId_5',
              'ProductId_ProductId_6', 'ProductId_ProductId_7', 'ProductId_ProductId_8',
              'ProductId_ProductId_9', 'ProductCategory_data_bundles', 'ProductCategory_financial_services',
              'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
              'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill',
              'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'CountryCode',
              'Amount', 'Value', 'PricingStrategy', 'FraudResult', 'Total_Transaction_Amount',
              'Average_Transaction_Amount', 'Transaction_Count', 'Std_Deviation_Transaction_Amount', 'Transaction_Hour',
              'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]
    
    X = data[['TotalRFMS', 'ProviderId_ProviderId_2', 'ProviderId_ProviderId_3',
              'ProviderId_ProviderId_4', 'ProviderId_ProviderId_5', 'ProviderId_ProviderId_6',
              'ProductId_ProductId_10', 'ProductId_ProductId_11', 'ProductId_ProductId_12',
              'ProductId_ProductId_13', 'ProductId_ProductId_14', 'ProductId_ProductId_15',
              'ProductId_ProductId_16', 'ProductId_ProductId_19', 'ProductId_ProductId_2',
              'ProductId_ProductId_20', 'ProductId_ProductId_21', 'ProductId_ProductId_22',
              'ProductId_ProductId_23', 'ProductId_ProductId_24', 'ProductId_ProductId_27',
              'ProductId_ProductId_3', 'ProductId_ProductId_4', 'ProductId_ProductId_5',
              'ProductId_ProductId_6', 'ProductId_ProductId_7', 'ProductId_ProductId_8',
              'ProductId_ProductId_9', 'ProductCategory_data_bundles', 'ProductCategory_financial_services',
              'ProductCategory_movies', 'ProductCategory_other', 'ProductCategory_ticket',
              'ProductCategory_transport', 'ProductCategory_tv', 'ProductCategory_utility_bill',
              'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'CountryCode',
              'Amount', 'Value', 'PricingStrategy', 'FraudResult', 'Total_Transaction_Amount',
              'Average_Transaction_Amount', 'Transaction_Count', 'Transaction_Hour',
              'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]
    y = data['RiskCategory'].map({'Good': 0, 'Bad': 1})  # Binary mapping for classification

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y


def modeling(X_train, X_test, y_train, y_test,results):
    # 1. Logistic Regression
    log_model = LogisticRegression(max_iter=1000, C=0.1, penalty='l2')
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]

    results['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, log_preds),
        'ROC AUC': roc_auc_score(y_test, log_probs),
        'Classification Report': classification_report(y_test, log_preds,zero_division=1)
    }
    # Save the model
    joblib.dump(log_model, 'model/logistic_regression_model.pkl')

    # 2. Random Forest
    rf_model = RandomForestClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_preds),
        'ROC AUC': roc_auc_score(y_test, rf_probs),
        'Classification Report': classification_report(y_test, rf_preds,zero_division=1)
    }
    # Save the model
    joblib.dump(rf_model, 'model/random_forest_model.pkl')

    # 3. Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)[:, 1]

    results['Decision Tree'] = {
        'Accuracy': accuracy_score(y_test, dt_preds),
        'ROC AUC': roc_auc_score(y_test, dt_probs),
        'Classification Report': classification_report(y_test, dt_preds, zero_division=1)
    }
    # Save the model
    joblib.dump(dt_model, 'model/decision_tree_model.pkl')

    # 4. Gradient Boosting Machine
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split=10,
                                          random_state=42)
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]

    results['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, gb_preds),
        'ROC AUC': roc_auc_score(y_test, gb_probs),
        'Classification Report': classification_report(y_test, gb_preds, zero_division=1)
    }

    # Save the model
    joblib.dump(gb_model, 'model/gradient_boosting_model.pkl')

    return results, log_preds, dt_preds, gb_preds, rf_preds
def model_metrics(results):
    # Displaying results for each model one by one
    for model in results.keys():
        # Logistic Regression
        metrics = results[model]
        print(f"Model: {model}")
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"ROC AUC: {metrics['ROC AUC']:.4f}")
        print("Classification Report:")
        print(metrics['Classification Report'])
        print("\n" + "="*50 + "\n")



def tune_models(X_train, y_train, X_test, y_test, search_method='grid', n_iter=10):
    param_grids = {
        'Gradient Boosting': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    }

    # Initialize the models
    models = {
        'Gradient Boosting': GradientBoostingClassifier()
    }

    # Choose the search method
    if search_method == 'grid':
        search_class = GridSearchCV
    elif search_method == 'random':
        search_class = RandomizedSearchCV
    else:
        raise ValueError("search_method should be 'grid' or 'random'")

    # Store the results for each model
    results = {}

    # Iterate through each model and perform hyperparameter tuning
    for model_name, model in models.items():
        print(f"Performing hyperparameter tuning for {model_name}...")

        # Select the appropriate search class
        param_grid = param_grids[model_name]

        # Perform hyperparameter tuning
        search = search_class(model, param_grid, cv=5, n_jobs=-1, n_iter=n_iter if search_method == 'random' else None, scoring='accuracy')
        search.fit(X_train, y_train)

        # Store the best model, parameters, and score
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # Evaluate on the test set
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Store the results in the dictionary
        results[model_name] = {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'classification_report': report
        }

        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best score for {model_name}: {best_score}")
        print(f"Classification report for {model_name}:\n{report}")

    return results


def save_best_model(results,X_train, y_train, X_test, y_test):
    best_params = {
        'subsample': 0.8,
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1
    }

    # Initialize the Gradient Boosting model with the best parameters
    gb_model = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)
    gb_probs = gb_model.predict_proba(X_test)[:, 1]

    results['Gradient Boosting'] = {
        'Accuracy': accuracy_score(y_test, gb_preds),
        'ROC AUC': roc_auc_score(y_test, gb_probs),
        'Classification Report': classification_report(y_test, gb_preds, zero_division=1)
    }

    # Save the model
    joblib.dump(gb_model, '../models/gradient_boosting_model.pkl')
    return gb_preds