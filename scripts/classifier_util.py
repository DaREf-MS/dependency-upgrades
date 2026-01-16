import pandas as pd
from scipy.stats import spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def load_classifiers() -> list:
    # Base estimators
    # rf_clf = RandomForestClassifier(random_state=42)
    et_clf = ExtraTreeClassifier(random_state=42)

    rnd_clf = RandomForestClassifier(random_state=42)
    ada_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42)
    xgb_clf = XGBClassifier(objective="binary:logistic", tree_method='hist', device='cuda', use_label_encoder=False, random_state=42)
    mlp_clf = MLPClassifier(random_state=42)

    ensemble_classifiers = [
        {
            "name": "ET",
            "classifier": et_clf
        },
        {
            "name": "RF",
            "classifier": rnd_clf
        },
        {
            "name": "XGBoost",
            "classifier": xgb_clf
        },
        {
            "name": "AdaBoost",
            "classifier": ada_clf
        },
        {
            "name": "MLP",
            "classifier": mlp_clf
        },
    ]

    return ensemble_classifiers


def correlation_analysis(X):
    # Calculate Spearman correlation matrix
    correlation_matrix, _ = spearmanr(X)

    highly_correlated_features = []
    columns = X.columns
    # Iterate through each pair of features
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            # If correlation is below 0.7, add both features
            corr = abs(correlation_matrix[i, j])
            if corr >= 0.7:
                # If correlation is above 0.7, select the feature with less importance score
                # print(columns[i], columns[j])
                col_i = columns[i]
                col_j = columns[j]
                # importance_i = metric_imp[col_i]
                # importance_j = metric_imp[col_j]
                # selected_feature = col_i if importance_i < importance_j else col_j
                # if selected_feature not in highly_correlated_features:
                # highly_correlated_features.append(selected_feature)
                highly_correlated_features.append(f"{col_i}, {col_j}")

    return highly_correlated_features


def calculate_r_squared(X, y):
    """
    Calculate R-squared value for a linear regression model.

    Parameters:
    X : array-like
        The independent variables.
    y : array-like
        The dependent variable.

    Returns:
    r_squared : float
        R-squared value of the linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    return r_squared


def redundancy_analysis(data):
    """
    Compute R-squared values for each feature predicting the remaining ones.

    Parameters:
    data : DataFrame
        The dataset containing all variables.

    Returns:
    r_squared_values : DataFrame
        DataFrame containing R-squared values for each feature predicting the remaining ones.
    """
    r_squared_values = {}

    for feature in data.columns:
        X = data.drop(columns=[feature])
        y = data[feature]
        r_squared = calculate_r_squared(X, y)
        r_squared_values[feature] = r_squared

    r_squared_df = pd.DataFrame(list(r_squared_values.items()), columns=["Feature", "R_squared"])
    r_squared_df = r_squared_df.loc[r_squared_df["R_squared"] >= 0.9, "Feature"].values.tolist()
    return r_squared_df