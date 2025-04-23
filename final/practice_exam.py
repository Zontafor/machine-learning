import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, r2_score

DATA_DIR = '/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/final/data/'
FIG_DIR = "/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/final/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- Set Torch Thread Count ---
torch.set_num_threads(14)

# --- Utility Functions ---
def load_csv(name):
    df = pd.read_csv(f"{DATA_DIR}{name}")
    df.columns = df.columns.str.strip()  # remove trailing spaces
    return df

def preprocess_demand_data(df):
    categorical = ['Device', 'Email', 'Payment', 'Region']
    return pd.get_dummies(df, columns=categorical, drop_first=True)

def split_train_test(df, target='HighSpend', split_index=700):
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    X_train, y_train = train.drop(columns=target), train[target]
    X_test, y_test = test.drop(columns=target), test[target]
    return X_train, X_test, y_train, y_test

# --- Section 1: Demand Prediction ---
demand_df = preprocess_demand_data(load_csv('demand.csv'))
X_train_raw, X_test_raw, y_train, y_test = split_train_test(demand_df)

# Feature scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

# Logistic Regression
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
y_pred_proba_lr = logreg.predict_proba(X_test)[:, 1]
print("Logistic Regression AUC:", roc_auc_score(y_test, y_pred_proba_lr))

# Odds ratio comparison for two customers
sample = X_test.iloc[0].copy()
X_features = sample.copy(); X_features['Time'] -= scaler.transform([[0]*X_train.shape[1]])[0][X_train.columns.get_loc('Time')] * 180 / 60  # scale adj
X_features['Region_Northeast'] = 1; X_features['Region_Midwest'] = 0
Y_features = sample.copy(); Y_features['Region_Northeast'] = 0; Y_features['Region_Midwest'] = 1

def compute_odds(features):
    aligned = pd.DataFrame([features])[X_train.columns]
    p = logreg.predict_proba(aligned)[0][1]
    return p / (1 - p)

print("Odds for X:", compute_odds(X_features))
print("Odds for Y:", compute_odds(Y_features))

# Decision Tree (depth=2)
dt2 = DecisionTreeClassifier(max_depth=2)
dt2.fit(X_train, y_train)
plt.figure(figsize=(12, 6))
plot_tree(dt2, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree (max_depth=2)")
plt.show()

# Cross-validated Decision Tree with parallelism
dt_cv = GridSearchCV(DecisionTreeClassifier(), {'min_samples_split': range(2, 51)}, cv=10, scoring='roc_auc', n_jobs=14)
dt_cv.fit(X_train, y_train)
best_tree = dt_cv.best_estimator_
y_pred_proba_tree = best_tree.predict_proba(X_test)[:, 1]
print("Best min_samples_split:", dt_cv.best_params_['min_samples_split'])
print("CV AUC:", dt_cv.best_score_)
print("Test AUC:", roc_auc_score(y_test, y_pred_proba_tree))

# Confusion matrix at threshold 0.1
y_pred_binary = (y_pred_proba_tree >= 0.1).astype(int)
cm = confusion_matrix(y_test, y_pred_binary)
TPR = cm[1,1] / (cm[1,1] + cm[1,0])
TNR = cm[0,0] / (cm[0,0] + cm[0,1])
print("Confusion Matrix:\n", cm)
print("True Positive Rate (TPR):", TPR)
print("True Negative Rate (TNR):", TNR)

# Save decision tree figure
plt.figure(figsize=(12, 6))
plot_tree(dt2, feature_names=X_train.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree (max_depth=2)")
plt.savefig(f"{FIG_DIR}/tree_depth.png")
plt.close()

# --- Section 2: Zeelow Linear Regression ---
zeelow_df = load_csv('zeelow.csv')
print("Zeelow columns:", zeelow_df.columns.tolist())

X = zeelow_df[['Size']]
y = zeelow_df['Prices']

# Simple Linear Regression
reg_all = LinearRegression().fit(X, y)
print("All Data R2:", r2_score(y, reg_all.predict(X)))
print("Slope:", reg_all.coef_[0])

# Clustering into Countries
zeelow_df['Cluster'] = KMeans(n_clusters=2, random_state=0).fit_predict(zeelow_df[['Size', 'Prices']])
grouped = zeelow_df.groupby('Cluster').agg({'Prices': 'mean', 'Size': 'mean'})
country_a, country_b = grouped['Prices'].idxmin(), grouped['Prices'].idxmax()

print("Country B Avg. Price:", grouped.loc[country_b, 'Prices'])
print("Country A Avg. Size:", grouped.loc[country_a, 'Size'])
print("Others count:", len(zeelow_df[~zeelow_df['Cluster'].isin([country_a, country_b])]))

# Regressions per country
Xa, ya = zeelow_df[zeelow_df['Cluster'] == country_a][['Size']], zeelow_df[zeelow_df['Cluster'] == country_a]['Prices']
Xb, yb = zeelow_df[zeelow_df['Cluster'] == country_b][['Size']], zeelow_df[zeelow_df['Cluster'] == country_b]['Prices']
rega = LinearRegression().fit(Xa, ya)
regb = LinearRegression().fit(Xb, yb)
print("Country A Slope:", rega.coef_[0], "R2:", r2_score(ya, rega.predict(Xa)))
print("Country B Slope:", regb.coef_[0])

# --- Section 3: EMNIST Dense Neural Net ---
with open(f"{DATA_DIR}emnist.p", 'rb') as f:
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = pickle.load(f)

X_train_tensor = torch.tensor(X_train_raw / 255.0, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_raw / 255.0, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_raw, dtype=torch.long)
y_test_tensor = torch.tensor(y_test_raw, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

class DenseNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 26)
        )
    def forward(self, x): return self.model(x)

model = DenseNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch.view(-1, 784))
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

model.eval()
total, correct = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch.view(-1, 784))
        _, preds = torch.max(output, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print("Dense NN Accuracy:", correct / total)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(14)

# File path and figure directory
DATA_PATH = "/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/final/data/emnist.p"
FIG_DIR = "/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/final/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load and preprocess data
with open(DATA_PATH, 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Normalize and reshape for CNN input (N, 1, 28, 28)
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0
X_train = X_train.reshape((-1, 1, 28, 28))
X_test = X_test.reshape((-1, 1, 28, 28))

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
X_test_t = torch.tensor(X_test)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Build dataset and dataloaders
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# CNN Model
class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # (28 -> 24 -> 12)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5) # (12 -> 8 -> 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model, loss, optimizer
model = EMNIST_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train model
EPOCHS = 10
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

# Save training curve
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
plt.title("CNN Training Loss (EMNIST)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{FIG_DIR}/cnn_training_curve.png")
plt.close()

# Evaluate accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        pred_labels = preds.argmax(dim=1)
        correct += (pred_labels == yb).sum().item()
        total += yb.size(0)

accuracy = correct / total
print(f"CNN Accuracy: {accuracy:.4f}")

# Parameter count breakdown
def count_params(model):
    total_params = 0
    print("\nTrainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total_params += count
            print(f"{name}: {count}")
    print(f"Total: {total_params}")

count_params(model)