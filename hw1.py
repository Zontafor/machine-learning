import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Ensure figure output directory exists
figure_dir = "/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw1/figures"
os.makedirs(figure_dir, exist_ok=True)

# Part A: Climate Change Analysis
# Load Climate Change Data
climate_data = pd.read_csv("/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw1/data/climate_change.csv")

# Split data into training (≤2002) and test (>2002)
train_data = climate_data[climate_data['Year'] <= 2002]
test_data = climate_data[climate_data['Year'] > 2002]

# Define full model features and reduced model features
features_full = ['CO2', 'CH4', 'N2O', 'CFC-11', 'CFC-12', 'Aerosols', 'TSI', 'MEI']
features_reduced = ['N2O', 'Aerosols', 'TSI', 'MEI']

# Extract train/test data for both models
X_train_full, X_test_full = train_data[features_full], test_data[features_full]
X_train_reduced, X_test_reduced = train_data[features_reduced], test_data[features_reduced]
y_train, y_test = train_data['Temp'], test_data['Temp']

# Standardize features
scaler = StandardScaler()
X_train_full = pd.DataFrame(scaler.fit_transform(X_train_full), columns=features_full)
X_test_full = pd.DataFrame(scaler.transform(X_test_full), columns=features_full)
X_train_reduced = pd.DataFrame(scaler.fit_transform(X_train_reduced), columns=features_reduced)
X_test_reduced = pd.DataFrame(scaler.transform(X_test_reduced), columns=features_reduced)

# Train Linear Regression Models for both full and reduced models
linreg_full = LinearRegression().fit(X_train_full, y_train)
linreg_reduced = LinearRegression().fit(X_train_reduced, y_train)

# Compute metrics for both models
metrics_full = {
    'Train R2': r2_score(y_train, linreg_full.predict(X_train_full)),
    'Test R2': r2_score(y_test, linreg_full.predict(X_test_full)),
    'Train MSE': mean_squared_error(y_train, linreg_full.predict(X_train_full)),
    'Test MSE': mean_squared_error(y_test, linreg_full.predict(X_test_full)),
    'Train MAE': mean_absolute_error(y_train, linreg_full.predict(X_train_full)),
    'Test MAE': mean_absolute_error(y_test, linreg_full.predict(X_test_full))
}

metrics_reduced = {
    'Train R2': r2_score(y_train, linreg_reduced.predict(X_train_reduced)),
    'Test R2': r2_score(y_test, linreg_reduced.predict(X_test_reduced)),
    'Train MSE': mean_squared_error(y_train, linreg_reduced.predict(X_train_reduced)),
    'Test MSE': mean_squared_error(y_test, linreg_reduced.predict(X_test_reduced)),
    'Train MAE': mean_absolute_error(y_train, linreg_reduced.predict(X_train_reduced)),
    'Test MAE': mean_absolute_error(y_test, linreg_reduced.predict(X_test_reduced))
}

# Extract N2O coefficient from both models
n2o_coef_full = linreg_full.coef_[features_full.index('N2O')]
n2o_coef_reduced = linreg_reduced.coef_[features_reduced.index('N2O')]
n2o_coefficients = {'Full Model': n2o_coef_full, 'Reduced Model': n2o_coef_reduced}

# Visualization: Predictions vs. Actual for Full Model
plt.figure()
plt.scatter(y_test, linreg_full.predict(X_test_full), label='Full Model', alpha=0.7)
plt.scatter(y_test, linreg_reduced.predict(X_test_reduced), label='Reduced Model', alpha=0.7, marker='x')
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Climate Change - Full vs. Reduced Model Predictions")
plt.legend()
plt.savefig(os.path.join(figure_dir, "climate_models_comparison.png"))
plt.show()

# Print Results
print("Climate Change - Full Model:", metrics_full)
print("Climate Change - Reduced Model:", metrics_reduced)
print("N2O Coefficients:", n2o_coefficients)

# Part B: Baseball Analytics
# Load Baseball Data
baseball_data = pd.read_csv("/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw1/data/baseball.csv")

# One-hot encoding categorical features
baseball_data = pd.get_dummies(baseball_data, columns=['League', 'Division', 'NewLeague'], drop_first=True)

# Split into training (70%) and test (30%) sets
train_baseball, test_baseball = train_test_split(baseball_data, test_size=0.3, random_state=42)
X_train_baseball = train_baseball.drop(columns=['Salary'])
y_train_baseball = train_baseball['Salary']
X_test_baseball = test_baseball.drop(columns=['Salary'])
y_test_baseball = test_baseball['Salary']

# Standardize baseball data
scaler_baseball = StandardScaler()
X_train_baseball = scaler_baseball.fit_transform(X_train_baseball)
X_test_baseball = scaler_baseball.transform(X_test_baseball)

# Ridge Regression with Cross-Validation
ridge_cv = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 10)}, cv=10)
ridge_cv.fit(X_train_baseball, y_train_baseball)
optimal_ridge = ridge_cv.best_estimator_
metrics_ridge = {
    'Train R2': r2_score(y_train_baseball, optimal_ridge.predict(X_train_baseball)),
    'Test R2': r2_score(y_test_baseball, optimal_ridge.predict(X_test_baseball)),
    'Train MSE': mean_squared_error(y_train_baseball, optimal_ridge.predict(X_train_baseball)),
    'Test MSE': mean_squared_error(y_test_baseball, optimal_ridge.predict(X_test_baseball)),
    'Train MAE': mean_absolute_error(y_train_baseball, optimal_ridge.predict(X_train_baseball)),
    'Test MAE': mean_absolute_error(y_test_baseball, optimal_ridge.predict(X_test_baseball))
}

# Print Baseball Results
print("Baseball - Ridge Regression (Best Alpha={}):".format(ridge_cv.best_params_['alpha']), metrics_ridge)
