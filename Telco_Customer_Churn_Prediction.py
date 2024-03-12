# Importing
##################################

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv(r"C:\Users\SERDAR\Documents\Mentorluk\DSMLBC14\Telco-Customer-Churn\Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)


##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

# GENERAL OVERVIEW
##################################

# Function to get a comprehensive overview of the dataset.
# It displays the dataset's shape, data types, head and tail, missing values, and quantile statistics.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)  # Displays the shape of the dataset
    print("##################### Types #####################")
    print(dataframe.dtypes)  # Displays data types of columns
    print("##################### Head #####################")
    print(dataframe.head(head))  # Displays the first few rows
    print("##################### Tail #####################")
    print(dataframe.tail(head))  # Displays the last few rows
    print("##################### NA #####################")
    print(dataframe.isnull().sum())  # Summarizes missing values
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  # Displays quantile statistics


check_df(df)  # Execute the function to display the dataset overview.


# CAPTURING NUMERICAL AND CATEGORICAL VARIABLES
##################################

# Function to identify categorical, numerical, and cardinal variables in the dataset.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    This function identifies the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numeric-looking categorical variables are also included in categorical variables.

    Parameters:
    dataframe: The dataframe to analyze.
    cat_th: Threshold for the number of unique values below which variables are considered categorical (default is 10).
    car_th: Threshold for the number of unique values above which variables are considered cardinal (default is 20).

    Returns:
    cat_cols: List of categorical variable names.
    num_cols: List of numerical variable names.
    cat_but_car: List of cardinal (categorical but with many unique values) variable names.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]  # Categorical columns
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtype != "O"]  # Numeric but categorical
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtype == "O"]  # Categorical but cardinal
    cat_cols += num_but_cat  # Combine categorical columns
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # Exclude cardinal columns from categorical columns

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]  # Numerical columns
    num_cols = [col for col in num_cols if col not in num_but_cat]  # Exclude numeric-looking categories

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


# Execute the function to identify and separate variable types.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ANALYSIS OF CATEGORICAL VARIABLES
##################################

# Function to summarize and optionally plot categorical variables.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# Loop through categorical columns to summarize them.
for col in cat_cols:
    cat_summary(df, col)


# ANALYSIS OF NUMERICAL VARIABLES
##################################

# Function to summarize and optionally plot numerical variables.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


# Loop through numerical columns to summarize and plot them.
for col in num_cols:
    num_summary(df, col, plot=True)


# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
##################################

# Function to compare the means of numerical columns by the target variable.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Execute the function for each numerical column.
for col in num_cols:
    target_summary_with_num(df, "Churn", col)


# ANALYSIS OF CATEGORICAL VARIABLES BY TARGET
##################################

# Function to summarize categorical variables by the target variable.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


# Execute the function for each categorical column.
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# CORRELATION ANALYSIS
##################################

# Display the correlation matrix for numerical variables.
df[num_cols].corr()

# Plot the correlation matrix.
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Analyze how each variable correlates with the target variable 'Churn'.
df.corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# TASK 2: FEATURE ENGINEERING
##################################

# MISSING VALUE ANALYSIS
##################################

# Check for any missing values in the dataset.
df.isnull().sum()


# Function to create a summary table of missing values in the dataframe.
def missing_values_table(dataframe, na_name=False):
    # Identify columns with missing values.
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # Count missing values and calculate their percentage.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Create a summary table.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    # Optionally return the names of columns with missing values.
    if na_name:
        return na_columns


# Execute the function to display the missing values table.
na_columns = missing_values_table(df, na_name=True)

# Fill missing values in 'TotalCharges' with the median of the column.
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Verify if missing values have been addressed.
df.isnull().sum()

# BASE MODEL SETUP
##################################

# Copy the original dataframe for model training to avoid altering the original data.
dff = df.copy()
# Exclude the target variable 'Churn' from the list of categorical columns.
cat_cols = [col for col in cat_cols if col not in ["Churn"]]


# Function to apply one-hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # Convert categorical variable into dummy/indicator variables.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Apply one-hot encoding to the dataframe.
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

# Define the target variable 'y' and features 'X'.
y = dff["Churn"]
X = dff.drop(["Churn", "customerID"], axis=1)

# List of models to train and evaluate.
models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

# Train each model and evaluate its performance using cross-validation.
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


# OUTLIER ANALYSIS
##################################

# Function to calculate outlier thresholds.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to check for outliers in a column.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Function to replace outliers with the thresholds.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Analyze and handle outliers for numerical columns.
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

##################################
# FEATURE EXTRACTION
##################################

# Creating yearly categorical variables from the 'tenure' feature
# to understand customer loyalty in terms of year ranges.
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Marking customers with 1 or 2-year contracts as 'Engaged' to identify potentially loyal customers.
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Identifying customers who do not have any of the support, backup, or protection services.
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
        x["TechSupport"] != "Yes") else 0, axis=1)

# Identifying young, not engaged (monthly contracts) customers potentially at higher risk of churning.
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0,
                                       axis=1)

# Counting the total number of services each customer has subscribed to.
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Identifying customers who have subscribed to any streaming service.
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

# Marking customers who use automatic payment methods.
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# Calculating the average monthly charges.
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Identifying the increase in current charges compared to the average charges.
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Calculating the average service fee per subscribed service.
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

# Viewing the head of the modified dataframe and its new shape after feature extraction.
df.head()
df.shape

##################################
# ENCODING
##################################

# Separating variable types for encoding.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# LABEL ENCODING: Transforming binary categorical variables into a machine-readable format.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# Identifying binary categorical columns for label encoding.
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

# Applying label encoding to binary columns.
for col in binary_cols:
    df = label_encoder(df, col)

# ONE-HOT ENCODING: Transforming categorical variables with more than two categories into a machine-readable format.
# Updating the list of categorical columns excluding the binary ones and the target variable 'Churn'.
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]


# Function to apply one-hot encoding to the specified categorical columns.
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Applying one-hot encoding to the dataframe.
df: DataFrame = one_hot_encoder(df, cat_cols, drop_first=True)

# Viewing the head of the dataframe after encoding.
df.head()
##################################
# MODELING
##################################

# Define the target variable 'y' and feature set 'X'.
y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)  # Drop 'customerID' as it's not a feature.

# Define a list of models to evaluate.
models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=12345)),
    ('RF', RandomForestClassifier(random_state=12345)),
    ('SVM', SVC(gamma='auto', random_state=12345)),
    ('XGB', XGBClassifier(random_state=12345)),
    ("LightGBM", LGBMClassifier(random_state=12345)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))
]

# Loop through models to train and evaluate them using cross-validation.
# Scoring metrics include accuracy, F1 score, ROC AUC, precision, and recall.
for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

################################################
# Random Forests Hyperparameter Tuning
################################################

# Initialize a Random Forest classifier.
rf_model = RandomForestClassifier(random_state=17)

# Define the parameter grid for Random Forests.
rf_params = {
    "max_depth": [5, 8, None],
    "max_features": [3, 5, 7, "auto"],
    "min_samples_split": [2, 5, 8, 15, 20],
    "n_estimators": [100, 200, 500]
}

# Perform grid search with cross-validation to find the best parameters.
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Output the best parameters and the best score achieved.
print(rf_best_grid.best_params_)
print(rf_best_grid.best_score_)

# Train the final model with the best parameters.
rf_final: object = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"Random Forests Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"Random Forests F1: {cv_results['test_f1'].mean()}")
print(f"Random Forests ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# XGBoost Hyperparameter Tuning
################################################

# Initialize an XGBoost classifier.
xgboost_model = XGBClassifier(random_state=17)

# Define the parameter grid for XGBoost.
xgboost_params = {
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [5, 8, 12, 15, 20],
    "n_estimators": [100, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}

# Perform grid search with cross-validation to find the best parameters.
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Train the final model with the best parameters.
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"XGBoost Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"XGBoost F1: {cv_results['test_f1'].mean()}")
print(f"XGBoost ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# LightGBM Hyperparameter Tuning
################################################

# Initialize a LightGBM classifier.
lgbm_model = LGBMClassifier(random_state=17)

# Define the parameter grid for LightGBM.
lgbm_params = {
    "learning_rate": [0.01, 0.1, 0.001],
    "n_estimators": [100, 300, 500, 1000],
    "colsample_bytree": [0.5, 0.7, 1]
}

# Perform grid search with cross-validation to find the best parameters.
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Train the final model with the best parameters.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the final model using cross-validation.
cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(f"LightGBM Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"LightGBM F1: {cv_results['test_f1'].mean()}")
print(f"LightGBM ROC AUC: {cv_results['test_roc_auc'].mean()}")

################################################
# CatBoost Optimization
################################################

# Initialize the CatBoost classifier with specific random state for reproducibility and silent mode to reduce log noise.
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Define the hyperparameter grid for the CatBoost model.
catboost_params = {
    "iterations": [200, 500],  # Number of trees
    "learning_rate": [0.01, 0.1],  # Step size for tree's weight adjustment
    "depth": [3, 6]  # Depth of trees
}

# Perform grid search to find the best hyperparameters within the defined grid, using cross-validation.
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Apply the best hyperparameters to the CatBoost model.
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Evaluate the optimized CatBoost model using cross-validation and several metrics.
cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])

# Print out the average scores from cross-validation to assess model performance.
print(f"CatBoost Accuracy: {cv_results['test_accuracy'].mean()}")
print(f"CatBoost F1 Score: {cv_results['test_f1'].mean()}")
print(f"CatBoost ROC AUC: {cv_results['test_roc_auc'].mean()}")


################################################
# Feature Importance Analysis
################################################

# Function to plot the importance of features for the given model.
def plot_importance(model, features, num=len(X), save=False):
    # Create a DataFrame with feature importances.
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    # Plotting the feature importances.
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # Optionally save the plot.
    if save:
        plt.savefig('importances.png')


# Plot feature importance for all final models to understand which features are most influential in predicting churn.
plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
