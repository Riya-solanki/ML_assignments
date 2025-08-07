import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_excel("C:/Users/hp/OneDrive - Amrita vishwa vidyapeetham/Documents/ML_assignments/lab_3/original (1).xlsx")
df.columns = df.columns.str.strip()

df['fluency'] = df['fluency'].str.lower().str.strip()
df['fluency'] = df['fluency'].replace({'hgih': 'high', 'mediu': 'medium', 'na': np.nan})
df['fluency'] = df['fluency'].fillna('unknown')

# Encode fluency column
le = LabelEncoder()
y_encoded = le.fit_transform(df['fluency'])
print("Encoded classes:", le.classes_)

# Encode embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
df['embedding'] = df['teacher'].apply(lambda x: model.encode(str(x)) if pd.notnull(x) else np.zeros(384))
X = np.vstack(df['embedding'].values)

# A1: Linear Regression with one attribute (using the first feature of embeddings)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_train_single = X_train[:, [0]]  # Select first feature of embeddings
X_test_single = X_test[:, [0]]

# Train the model
reg_single = LinearRegression().fit(X_train_single, y_train)

# Predictions
y_train_pred_single = reg_single.predict(X_train_single)
y_test_pred_single = reg_single.predict(X_test_single)

# A2: Calculate metrics for single attribute
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Metrics for training set (single attribute)
mse_train_single, rmse_train_single, mape_train_single, r2_train_single = calculate_metrics(y_train, y_train_pred_single)
print("Single Attribute - Training Set Metrics:")
print(f"MSE: {mse_train_single:.4f}")
print(f"RMSE: {rmse_train_single:.4f}") 
print(f"MAPE: {mape_train_single:.4f}%")
print(f"R2: {r2_train_single:.4f}")


# Metrics for test set (single attribute)
mse_test_single, rmse_test_single, mape_test_single, r2_test_single = calculate_metrics(y_test, y_test_pred_single)
print("\nSingle Attribute - Test Set Metrics:")
print(f"MSE: {mse_test_single:.4f}")
print(f"RMSE: {rmse_test_single:.4f}")
print(f"MAPE: {mape_test_single:.4f}%")
print(f"R2: {r2_test_single:.4f}")

# A3: Linear Regression with all attributes
reg_all = LinearRegression().fit(X_train, y_train)

# Predictions
y_train_pred_all = reg_all.predict(X_train)
y_test_pred_all = reg_all.predict(X_test)

# Metrics for training set (all attributes)
mse_train_all, rmse_train_all, mape_train_all, r2_train_all = calculate_metrics(y_train, y_train_pred_all)
print("\nAll Attributes - Training Set Metrics:")
print(f"MSE: {mse_train_all:.4f}")
print(f" RMSE: {rmse_train_all:.4f}")
print(f" MAPE: {mape_train_all:.4f}%")
print(f"R2: {r2_train_all:.4f}")

# Metrics for test set (all attributes)
mse_test_all, rmse_test_all, mape_test_all, r2_test_all = calculate_metrics(y_test, y_test_pred_all)
print("\nAll Attributes - Test Set Metrics:")
print(f"MSE: {mse_test_all:.4f}")
print(f"RMSE: {rmse_test_all:.4f}")
print(f" MAPE: {mape_test_all:.4f}%")
print(f"R2: {r2_test_all:.4f}")

# A4: K-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)
print("\nK-means Clustering (k=2) Labels:", kmeans.labels_)
print("K-means Clustering (k=2) Cluster Centers Shape:", kmeans.cluster_centers_.shape)

# A5: Calculate Silhouette Score, CH Score, and DB Index for k=2
sil_score = silhouette_score(X_train, kmeans.labels_)
ch_score = calinski_harabasz_score(X_train, kmeans.labels_)
db_score = davies_bouldin_score(X_train, kmeans.labels_)
print("\nClustering Metrics for k=2:")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")

# A6: K-means clustering for different k values and evaluate metrics
k_values = range(2, 20)
sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    sil_scores.append(silhouette_score(X_train, kmeans.labels_))
    ch_scores.append(calinski_harabasz_score(X_train, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X_train, kmeans.labels_))

# Plotting metrics against k values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o')
plt.title('Silhouette Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title('Calinski-Harabasz Score vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('CH Score')

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('DB Index')

plt.tight_layout()
plt.show()

# A7: Elbow plot to determine optimal k
distortions = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_values, distortions, marker='o')
plt.title('Elbow Plot for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.show()