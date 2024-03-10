# Telco Customer Churn Prediction

## Project Overview
This project aims to develop a machine learning model capable of predicting which customers are likely to churn from a fictional telecom company. Prior to model development, it's essential to perform data analysis and feature engineering steps to prepare the data adequately.

## Dataset Description
The dataset encompasses information from 7043 customers in California during the third quarter, detailing their use of home phone and Internet services provided by the telecom company. It includes 21 variables, reflecting customer demographics, services subscribed, account information, and the churn status.

### Features
- `CustomerId`: Unique identifier for the customer.
- `Gender`: Customer's gender (Male, Female).
- `SeniorCitizen`: Indicates if the customer is a senior citizen (1, 0).
- `Partner`: Indicates if the customer has a partner (Yes, No).
- `Dependents`: Indicates if the customer has dependents (Yes, No).
- `Tenure`: Number of months the customer has been with the company.
- `PhoneService`: Indicates if the customer has phone service (Yes, No).
- `MultipleLines`: Indicates if the customer has multiple lines (Yes, No, No phone service).
- `InternetService`: Customer's internet service provider (DSL, Fiber optic, No).
- `OnlineSecurity`: Indicates if the customer has online security (Yes, No, No internet service).
- `OnlineBackup`: Indicates if the customer has online backup (Yes, No, No internet service).
- `DeviceProtection`: Indicates if the customer has device protection (Yes, No, No internet service).
- `TechSupport`: Indicates if the customer has tech support (Yes, No, No internet service).
- `StreamingTV`: Indicates if the customer has streaming TV (Yes, No, No internet service).
- `StreamingMovies`: Indicates if the customer has streaming movies (Yes, No, No internet service).
- `Contract`: The contract term of the customer (Month-to-month, One year, Two year).
- `PaperlessBilling`: Indicates if the customer has paperless billing (Yes, No).
- `PaymentMethod`: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TotalCharges`: The total amount charged to the customer.
- `Churn`: Whether the customer churned (Yes or No).

## Methodology
The analysis and prediction process involves several key steps:

1. **Exploratory Data Analysis (EDA):** Examining the overall dataset, identifying numerical and categorical variables, and analyzing their distribution and relationship with the target variable.

2. **Feature Engineering:**
    - Handling missing and outlier values.
    - Creating new features to better capture customer behavior.
    - Encoding categorical variables.
    - Standardizing numerical variables.

3. **Model Building:**
    - Training various machine learning models like Logistic Regression, KNN, Decision Trees, Random Forest, XGBoost, LightGBM, and CatBoost.
    - Comparing their performance to select the best model.

4. **Model Optimization:** Tuning the selected model to improve its predictive performance.

5. **Feature Importance Analysis:** Identifying and interpreting the most influential features affecting customer churn.

## Key Insights
- Detailed insights from EDA, such as the impact of contract type and payment method on churn rate.
- Importance of feature engineering in improving model performance.
- Performance metrics of different models and rationale behind selecting the best model.

## How to Run
- Ensure you have Python and necessary libraries installed.
- Clone the repository and navigate to the project directory.
- Run the script to perform EDA, feature engineering, and model training:
python telco_customer_churn_prediction.py

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- LightGBM
- XGBoost

## Future Directions
- Further exploration of advanced feature engineering techniques.
- Testing additional models and ensemble methods.
- Deployment of the model for real-time churn prediction.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- Recognition of the fictional telecom company for providing a comprehensive dataset.
- Gratitude towards the Python and machine learning communities for their valuable resources.
