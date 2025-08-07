import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

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

# A1: Train-test split and kNN classifier
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Training metrics
y_train_pred = knn.predict(X_train)
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=[str(c) for c in le.classes_], zero_division=0))
print("Training Confusion Matrix:")
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
print(train_conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(train_conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Training Confusion Matrix')
plt.show()

# Test metrics
y_test_pred = knn.predict(X_test)
unique_test_classes = np.unique(y_test)
target_names = [str(le.classes_[i]) for i in unique_test_classes]
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=target_names, zero_division=0))
print("Test Confusion Matrix:")
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
print(test_conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()

# Model fit analysis
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
if train_accuracy > test_accuracy + 0.1:
    print("Model is likely overfitting.")
elif train_accuracy < 0.6 and test_accuracy < 0.6:
    print("Model is likely underfitting.")
else:
    print("Model appears to be regular fitting.")

# A2: Price prediction (synthetic example)
np.random.seed(42)
X_price = np.random.rand(100, 2) * 10
y_price = X_price[:, 0] * 2 + X_price[:, 1] * 3 + np.random.randn(100) * 0.5
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.3, random_state=42)
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train_price, y_train_price)
y_train_pred_price = knn_reg.predict(X_train_price)
y_test_pred_price = knn_reg.predict(X_test_price)

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

train_mse, train_rmse, train_mape, train_r2 = calculate_regression_metrics(y_train_price, y_train_pred_price)
print("Training Metrics for Price Prediction:")
print(f"MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%, R²: {train_r2:.2f}")
test_mse, test_rmse, test_mape, test_r2 = calculate_regression_metrics(y_test_price, y_test_pred_price)
print("Test Metrics for Price Prediction:")
print(f"MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%, R²: {test_r2:.2f}")

# A3: Synthetic 2D data
X_2d = np.random.uniform(1, 10, (20, 2))
y_2d = (X_2d[:, 0] + X_2d[:, 1] > 10).astype(int)
plt.figure(figsize=(6, 6))
plt.scatter(X_2d[y_2d == 0][:, 0], X_2d[y_2d == 0][:, 1], c='blue', label='Class 0 (Blue)', alpha=0.6)
plt.scatter(X_2d[y_2d == 1][:, 0], X_2d[y_2d == 1][:, 1], c='red', label='Class 1 (Red)', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training Data: 20 Points (2 Classes)')
plt.legend()
plt.grid(True)
plt.show()

# A4: Test set classification (k=3)
x_range = np.arange(0, 10.1, 0.1)
X_test_2d = np.array(np.meshgrid(x_range, x_range)).T.reshape(-1, 2)
knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_2d, y_2d)
y_test_2d_pred = knn_2d.predict(X_test_2d)
plt.figure(figsize=(6, 6))
plt.scatter(X_test_2d[y_test_2d_pred == 0][:, 0], X_test_2d[y_test_2d_pred == 0][:, 1], c='blue', label='Predicted Class 0 (Blue)', alpha=0.1, s=10)
plt.scatter(X_test_2d[y_test_2d_pred == 1][:, 0], X_test_2d[y_test_2d_pred == 1][:, 1], c='red', label='Predicted Class 1 (Red)', alpha=0.1, s=10)
plt.scatter(X_2d[y_2d == 0][:, 0], X_2d[y_2d == 0][:, 1], c='blue', edgecolors='black', label='Training Class 0', s=100)
plt.scatter(X_2d[y_2d == 1][:, 0], X_2d[y_2d == 1][:, 1], c='red', edgecolors='black', label='Training Class 1', s=100)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data Predictions (k=3)')
plt.legend()
plt.grid(True)
plt.show()

# A5: Vary k
k_values = [1, 5, 7, 9]
for k in k_values:
    knn_2d = KNeighborsClassifier(n_neighbors=k)
    knn_2d.fit(X_2d, y_2d)
    y_test_2d_pred = knn_2d.predict(X_test_2d)
    plt.figure(figsize=(6, 6))
    plt.scatter(X_test_2d[y_test_2d_pred == 0][:, 0], X_test_2d[y_test_2d_pred == 0][:, 1], c='blue', label='Predicted Class 0 (Blue)', alpha=0.1, s=10)
    plt.scatter(X_test_2d[y_test_2d_pred == 1][:, 0], X_test_2d[y_test_2d_pred == 1][:, 1], c='red', label='Predicted Class 1 (Red)', alpha=0.1, s=10)
    plt.scatter(X_2d[y_2d == 0][:, 0], X_2d[y_2d == 0][:, 1], c='blue', edgecolors='black', label='Training Class 0', s=100)
    plt.scatter(X_2d[y_2d == 1][:, 0], X_2d[y_2d == 1][:, 1], c='red', edgecolors='black', label='Training Class 1', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Test Data Predictions (k={k})')
    plt.legend()
    plt.grid(True)
    plt.show()

# A7: Hyper-parameter tuning
param_grid = {'n_neighbors': list(range(1, 12))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
best_knn = grid_search.best_estimator_
y_test_pred_best = best_knn.predict(X_test)
print("Test Classification Report (Best k):")
print(classification_report(y_test, y_test_pred_best, target_names=target_names, zero_division=0))