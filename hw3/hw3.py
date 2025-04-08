import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering

### ---- Part A1 ---- ###

# Load the data
X, y = pickle.load(open('/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw3/data/listens.p', 'rb'))

# (a) Number of users and songs
num_users, num_songs = X.shape
print(f"(a) Number of users: {num_users}, Number of songs: {num_songs}")

# (b) Max listens to a single song by a user
max_listens = np.max(X)
print(f"(b) Max number of listens to a single song by a user: {max_listens}")

# (c) % of users who listened to the special song
percent_listened = 100 * np.mean(y)
print(f"(c) Percentage of users who listened to the special song: {percent_listened:.2f}%")

# (d) Total listens per song
total_listens_per_song = np.sum(X, axis=0)
print(f"(d) Total song listens - min: {np.min(total_listens_per_song)}, max: {np.max(total_listens_per_song)}, avg: {np.mean(total_listens_per_song):.2f}")

# (e) Number of users who listened to each song
users_per_song = np.sum(X > 0, axis=0)
print(f"(e) Users per song - min: {np.min(users_per_song)}, max: {np.max(users_per_song)}, avg: {np.mean(users_per_song):.2f}")

# (f) Total listens per user
total_listens_per_user = np.sum(X, axis=1)
print(f"(f) Listens per user - min: {np.min(total_listens_per_user)}, max: {np.max(total_listens_per_user)}, avg: {np.mean(total_listens_per_user):.2f}")

# (g) Number of songs listened to by each user
songs_per_user = np.sum(X > 0, axis=1)
print(f"(g) Songs per user - min: {np.min(songs_per_user)}, max: {np.max(songs_per_user)}, avg: {np.mean(songs_per_user):.2f}")

### ---- Part A2 ---- ###

# Load the data
X, y = pickle.load(open('/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw3/data/listens.p', 'rb'))

# Split: first 1400 rows = training, rest = test
X_train, X_test = X[:1400], X[1400:]
y_train, y_test = y[:1400], y[1400:]

### ---- A.2(a) ---- ###
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print("\nA.2(a) Random Forest (default):")
print(f"Out-of-sample AUC: {rf_auc:.4f}")

### ---- A.2(b) ---- ###
from sklearn.model_selection import StratifiedKFold

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric': ['euclidean', 'manhattan', 'hamming']
}

# Initialize best values
best_auc = 0
best_params = None
best_model = None

# Try each combination manually with GridSearchCV
for metric in param_grid['metric']:
    knn = KNeighborsClassifier(metric=metric)
    gs = GridSearchCV(knn, {'n_neighbors': param_grid['n_neighbors']}, cv=5, scoring='roc_auc')
    gs.fit(X_train, y_train)
    
    # Evaluate on test set
    probs = gs.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"\nA.2(b) k-NN (metric: {metric})")
    print(f"Best k: {gs.best_params_['n_neighbors']}, Test AUC: {auc:.4f}")
    
    # Track best overall model
    if auc > best_auc:
        best_auc = auc
        best_params = {'k': gs.best_params_['n_neighbors'], 'metric': metric}
        best_model = gs.best_estimator_

print(f"\nBest k-NN Model Overall:")
print(f"Metric: {best_params['metric']}, k: {best_params['k']}, AUC: {best_auc:.4f}")

### ---- A.3 ---- ###

print("\n--- A.3: Column Normalization + k-NN ---")

best_auc_norm = 0
best_params_norm = None
best_model_norm = None

for metric in param_grid['metric']:
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('knn', KNeighborsClassifier(metric=metric))
    ])
    
    gs = GridSearchCV(pipe, {'knn__n_neighbors': param_grid['n_neighbors']}, cv=5, scoring='roc_auc')
    gs.fit(X_train, y_train)
    
    probs = gs.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print(f"\nA.3 k-NN + Normalization (metric: {metric})")
    print(f"Best k: {gs.best_params_['knn__n_neighbors']}, Test AUC: {auc:.4f}")
    
    if auc > best_auc_norm:
        best_auc_norm = auc
        best_params_norm = {'k': gs.best_params_['knn__n_neighbors'], 'metric': metric}
        best_model_norm = gs.best_estimator_

print(f"\nBest Normalized k-NN Model:")
print(f"Metric: {best_params_norm['metric']}, k: {best_params_norm['k']}, AUC: {best_auc_norm:.4f}")

### ---- A.4 ---- ###
print("\n--- A.4: PCA + k-NN ---")

pca_components = [10, 20, 30, 50]
best_auc_pca = 0
best_params_pca = None
best_model_pca = None

for metric in param_grid['metric']:
    pipe = Pipeline([
        ('pca', PCA()),
        ('knn', KNeighborsClassifier(metric=metric))
    ])

    gs = GridSearchCV(pipe, {
        'pca__n_components': pca_components,
        'knn__n_neighbors': param_grid['n_neighbors']
    }, cv=5, scoring='roc_auc')

    gs.fit(X_train, y_train)
    probs = gs.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"\nA.4 k-NN + PCA (metric: {metric})")
    print(f"Best PCA components: {gs.best_params_['pca__n_components']}, Best k: {gs.best_params_['knn__n_neighbors']}, Test AUC: {auc:.4f}")

    if auc > best_auc_pca:
        best_auc_pca = auc
        best_params_pca = {
            'metric': metric,
            'k': gs.best_params_['knn__n_neighbors'],
            'n_components': gs.best_params_['pca__n_components']
        }
        best_model_pca = gs.best_estimator_

print(f"\nBest PCA + k-NN Model:")
print(f"Metric: {best_params_pca['metric']}, k: {best_params_pca['k']}, n_components: {best_params_pca['n_components']}, AUC: {best_auc_pca:.4f}")

### ---- A.5 ---- ###
print("\n--- A.5: PCA + Normalization + k-NN ---")

best_auc_pca_norm = 0
best_params_pca_norm = None
best_model_pca_norm = None

for metric in param_grid['metric']:
    pipe = Pipeline([
        ('pca', PCA()),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(metric=metric))
    ])

    gs = GridSearchCV(pipe, {
        'pca__n_components': pca_components,
        'knn__n_neighbors': param_grid['n_neighbors']
    }, cv=5, scoring='roc_auc')

    gs.fit(X_train, y_train)
    probs = gs.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"\nA.5 PCA + Normalization + k-NN (metric: {metric})")
    print(f"Best PCA components: {gs.best_params_['pca__n_components']}, Best k: {gs.best_params_['knn__n_neighbors']}, Test AUC: {auc:.4f}")

    if auc > best_auc_pca_norm:
        best_auc_pca_norm = auc
        best_params_pca_norm = {
            'metric': metric,
            'k': gs.best_params_['knn__n_neighbors'],
            'n_components': gs.best_params_['pca__n_components']
        }
        best_model_pca_norm = gs.best_estimator_

print(f"\nBest PCA + Normalization + k-NN Model:")
print(f"Metric: {best_params_pca_norm['metric']}, k: {best_params_pca_norm['k']}, n_components: {best_params_pca_norm['n_components']}, AUC: {best_auc_pca_norm:.4f}")

### ---- Part B ---- ###

# Load the returns data
df = pd.read_csv('/Users/mlwu/Documents/CMU/Machine Learning Fundamentals/homework/hw3/data/returns.csv')

# Quick look
print(df.shape)
print(df.columns[:10])  # first few columns
print(df[['symbol', 'Industry']].head())

### ---- B.1(a) ---- ###
consumer_count = (df['Industry'] == 'Consumer Discretionary').sum()
energy_count = (df['Industry'] == 'Energy').sum()

print(f"B.1(a): Consumer Discretionary: {consumer_count} companies")
print(f"        Energy: {energy_count} companies")

### ---- B.1(b) ---- ###
date_cols = [col for col in df.columns if col.startswith('avg') and '2008' <= col[3:7] <= '2010']

# Compute average return per sector over time
sector_returns = df.groupby('Industry')[date_cols].mean().T  # Transpose for plotting

# Select a few sectors to plot
selected_sectors = ['Consumer Discretionary', 'Energy', 'Health Care', 'Financials']
plt.figure(figsize=(12, 6))

for sector in selected_sectors:
    plt.plot(sector_returns.index, sector_returns[sector], label=sector)

plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylabel('Average Monthly Return')
plt.title('Sector Returns (2008–2010)')
plt.legend()
plt.tight_layout()
plt.show()

# Drop non-numeric columns to get time series
return_cols = [col for col in df.columns if col.startswith("avg")]
returns = df[return_cols].copy()

### Standardize the time series
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

### ---- B.2(a) ---- ###
kmeans_4 = KMeans(n_clusters=4, random_state=42)
df['cluster_4'] = kmeans_4.fit_predict(returns_scaled)

# Plot centroids
plt.figure(figsize=(12, 6))
for i, centroid in enumerate(kmeans_4.cluster_centers_):
    plt.plot(return_cols, centroid, label=f'Cluster {i}')
plt.xticks(rotation=45)
plt.title("B.2(a) k-Means Cluster Centroids (k=4)")
plt.ylabel("Standardized Return")
plt.legend()
plt.tight_layout()
plt.show()

### ---- B.2(b) ---- ###
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(returns_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("B.2(b) Scree Plot for k-Means")
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

### ---- B.2(c) ---- ###
# Visually inspect the scree plot to decide \\ k = 4

### ---- B.2(d) ---- ###
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(returns_scaled)

# Count companies per cluster
cluster_sizes = df['cluster'].value_counts().sort_index()
print(f"\nB.2(d): Companies per cluster:\n{cluster_sizes}")

# Industry breakdown per cluster
industry_by_cluster = df.groupby(['cluster', 'Industry']).size().unstack(fill_value=0)
print("\nIndustry breakdown per cluster:")
print(industry_by_cluster)

### ---- B.2(e) ---- ###
oct_2008 = df.groupby('cluster')['avg200810'].mean()
mar_2009 = df.groupby('cluster')['avg200903'].mean()

print("\nB.2(e): Avg Return per Cluster")
print(f"October 2008 (worst month):\n{oct_2008}")
print(f"March 2009 (rebound month):\n{mar_2009}")

### ---- B.3(c) ---- ###

# Compute correlation matrix of time series (companies as rows)
correlation_matrix = np.corrcoef(returns_scaled)
similarity_matrix = (correlation_matrix + 1) / 2  # Shift to [0, 1]

# Spectral clustering
spectral = SpectralClustering(
    n_clusters=4,
    affinity='precomputed',
    random_state=42,
    assign_labels='kmeans'
)

df['spectral_cluster'] = spectral.fit_predict(similarity_matrix)

# Cluster sizes
print("\nSpectral Clustering – Cluster Counts:")
print(df['spectral_cluster'].value_counts())

# Compare with k-means clusters
comparison = pd.crosstab(df['cluster'], df['spectral_cluster'], rownames=['kMeans'], colnames=['Spectral'])
print("\nk-Means vs Spectral Clustering Cross-tab:")
print(comparison)