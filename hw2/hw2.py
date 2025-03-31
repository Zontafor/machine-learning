import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# PART A: Titanic
# ------------------------------

# A.1 Load Titanic dataset and preprocess
try:
    titanic_df = pd.read_csv("/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw2/data/titanic.csv")
except FileNotFoundError:
    raise FileNotFoundError("Make sure 'titanic.csv' is in the same directory as this script.")

# Select relevant features
titanic_features = titanic_df[["Pclass", "Sex", "SibSp", "ParCh", "Fare", "Embarked", "Survived"]].copy()

# One-hot encode 'Sex' and 'Embarked', dropping first to avoid multicollinearity
titanic_features = pd.get_dummies(titanic_features, columns=["Sex", "Embarked"], drop_first=True)

# Define features and target
X_titanic = titanic_features.drop("Survived", axis=1)
y_titanic = titanic_features["Survived"]

# Fit unregularized logistic regression
logreg_A1 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
logreg_A1.fit(X_titanic, y_titanic)

# Report intercept and coefficients
print("\n--- A.1 Logistic Regression Coefficients ---")
print("Intercept:", logreg_A1.intercept_[0])
coef_df = pd.DataFrame({"Feature": X_titanic.columns, "Coefficient": logreg_A1.coef_[0]})
print(coef_df)

# A.3 Predict survival probability for a specific passenger
# Female, 1st class, Fare = 62.50, Spouse=1, No siblings, ParCh = 2, Embarked=Q
passenger_data = pd.DataFrame({
    'Pclass': [1],
    'SibSp': [1],
    'ParCh': [2],
    'Fare': [62.5],
    'Sex_male': [0],
    'Embarked_Q': [1],
    'Embarked_S': [0]
})
try:
    pred_prob = logreg_A1.predict_proba(passenger_data)[0, 1]
    print("\n--- A.3 Predicted survival probability for specified passenger ---")
    print(f"Probability of survival: {pred_prob:.4f}")
except Exception as e:
    print("Error predicting passenger survival:", str(e))

# A.4 Treat ParCh as categorical
parch_encoded_df = pd.get_dummies(
    titanic_df[["Pclass", "Sex", "SibSp", "ParCh", "Fare", "Embarked", "Survived"]],
    columns=["Sex", "Embarked", "ParCh"], drop_first=True
)
X_A4 = parch_encoded_df.drop("Survived", axis=1)
y_A4 = parch_encoded_df["Survived"]
logreg_A4 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
logreg_A4.fit(X_A4, y_A4)

print("\n--- A.4 Coefficients for ParCh dummies ---")
for col, coef in zip(X_A4.columns, logreg_A4.coef_[0]):
    if "ParCh_" in col:
        print(f"{col}: {coef:.4f}")

# Frequency of each ParCh value
print("\nParCh class counts:")
print(titanic_df["ParCh"].value_counts().sort_index())

# ------------------------------
# PART B: Framingham
# ------------------------------

# Load and clean Framingham dataset
try:
    framingham_df = pd.read_csv("/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw2/data/framingham.csv")
except FileNotFoundError:
    raise FileNotFoundError("Make sure 'framingham.csv' is in the same directory as this script.")

# Drop rows with missing values (could improve with imputation)
framingham_df.dropna(inplace=True)

# Separate features and target
X_fram = pd.get_dummies(framingham_df.drop("TenYearCHD", axis=1), columns=["Education"], drop_first=True)
y_fram = framingham_df["TenYearCHD"]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_fram, y_fram, test_size=0.3, stratify=y_fram, random_state=2023
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# B.1(a) Unregularized Logistic Regression
model_b1a = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
model_b1a.fit(X_train_scaled, y_train)
preds_b1a = model_b1a.predict_proba(X_test_scaled)[:, 1]
auc_b1a = roc_auc_score(y_test, preds_b1a)
print("\n--- B.1(a) Unregularized Logistic Regression AUC ---")
print(f"Test AUC: {auc_b1a:.4f}")

# B.1(b) L1-Regularized Logistic Regression with CV
model_b1b = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10, 100],
    penalty='l1',
    solver='liblinear',
    cv=5,
    scoring='roc_auc',
    random_state=2023,
    max_iter=1000
)
model_b1b.fit(X_train_scaled, y_train)
preds_b1b = model_b1b.predict_proba(X_test_scaled)[:, 1]
auc_b1b = roc_auc_score(y_test, preds_b1b)
print("\n--- B.1(b) L1-Regularized Logistic Regression AUC ---")
print(f"Test AUC: {auc_b1b:.4f}")

# B.1(c) L2-Regularized Logistic Regression with CV
model_b1c = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10, 100],
    penalty='l2',
    solver='liblinear',
    cv=5,
    scoring='roc_auc',
    random_state=2023,
    max_iter=1000
)
model_b1c.fit(X_train_scaled, y_train)
preds_b1c = model_b1c.predict_proba(X_test_scaled)[:, 1]
auc_b1c = roc_auc_score(y_test, preds_b1c)
print("\n--- B.1(c) L2-Regularized Logistic Regression AUC ---")
print(f"Test AUC: {auc_b1c:.4f}")

# B.1(d) Decision Tree with tuning
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=2023),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
dt_grid.fit(X_train, y_train)

dt_best = dt_grid.best_estimator_
preds_b1d = dt_best.predict_proba(X_test)[:, 1]
auc_b1d = roc_auc_score(y_test, preds_b1d)
print("\n--- B.1(d) Decision Tree AUC (tuned) ---")
print(f"Best Params: {dt_grid.best_params_}")
print(f"Test AUC: {auc_b1d:.4f}")

# ------------------------------
# B.2 Simple Decision Tree (≤ 3 queries)
# ------------------------------

# B.2(a) Train simple decision tree
simple_tree = DecisionTreeClassifier(max_depth=2, random_state=2023)
simple_tree.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(12, 6))
plot_tree(simple_tree, feature_names=X_train.columns, class_names=['No CHD', 'CHD'], filled=True)
plt.title("B.2(a) Simple Decision Tree (max_depth=2)")
plt.show()

# B.2(b) Report AUC
simple_preds = simple_tree.predict_proba(X_test)[:, 1]
simple_auc = roc_auc_score(y_test, simple_preds)
print("\n--- B.2(b) Simple Tree Test AUC ---")
print(f"Test AUC: {simple_auc:.4f}")

# B.2(c) Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, simple_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Simple Tree (AUC = {simple_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate (1 - TNR)")
plt.ylabel("True Positive Rate")
plt.title("B.2(c) ROC Curve - Simple Tree")
plt.legend()
plt.grid(True)
plt.show()

# B.2(d) Max TPR when TNR >= 60%
tnr = 1 - fpr
valid_indices = np.where(tnr >= 0.60)[0]
if len(valid_indices) > 0:
    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
    best_tpr = tpr[best_idx]
    best_threshold = thresholds[best_idx]
    print("\n--- B.2(d) Threshold Analysis ---")
    print(f"Max TPR with TNR ≥ 60%: {best_tpr:.4f} at threshold = {best_threshold:.4f}")
else:
    print("\n--- B.2(d) Threshold Analysis ---")
    print("No threshold found where TNR ≥ 60%.")