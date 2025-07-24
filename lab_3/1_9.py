import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_excel("C:/Users/hp/OneDrive - Amrita vishwa vidyapeetham/Documents/ML_assignments/lab_3/original (1).xlsx")
#print(df.head())
# Clean column names
df.columns = df.columns.str.strip()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode teacher answers safely
df['embedding'] = df['teacher'].apply(lambda x: model.encode(str(x)) if pd.notnull(x) else np.zeros(384))
df['student_embedding'] = df['student'].apply(lambda x: model.encode(str(x)) if pd.notnull(x) else np.zeros(384))

# Convert to NumPy arrays
teacher_vectors = np.vstack(df['embedding'].values)
student_vectors = np.vstack(df['student_embedding'].values)

# Compute centroids (mean vectors)
teacher_centroid = np.mean(teacher_vectors, axis=0)
student_centroid = np.mean(student_vectors, axis=0)

# Compute intra-class spread (standard deviation)
teacher_spread = np.std(teacher_vectors, axis=0)
student_spread = np.std(student_vectors, axis=0)

# Overall spread as scalar (Euclidean norm of std vector)
teacher_spread_scalar = np.linalg.norm(teacher_spread)
student_spread_scalar = np.linalg.norm(student_spread)

# Inter-class distance (Euclidean distance between centroids)
interclass_distance = np.linalg.norm(teacher_centroid - student_centroid)

# Print results
print("Intra-class Spread (Teacher):", teacher_spread_scalar)
print("Intra-class Spread (Student):", student_spread_scalar)
print("Inter-class Distance:", interclass_distance)

#%%
import matplotlib.pyplot as plt

# Select one dimension (e.g., the first dimension from teacher embedding)
feature_data = teacher_vectors[:, 0]  # taking 1st feature across all teacher answers

# Histogram
plt.figure(figsize=(8,5))
plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Embedding Feature 0 (Teacher Answers)")
plt.xlabel("Feature Value (Dimension 0)")

plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Calculate mean and variance
mean_val = np.mean(feature_data)
variance_val = np.var(feature_data)

print("Mean of Feature 0:", mean_val)
print("Variance of Feature 0:", variance_val)

#%%
from scipy.spatial.distance import minkowski

vec1 = teacher_vectors[0]
vec2 = teacher_vectors[1]

r_values = list(range(1, 11))
distances = [minkowski(vec1, vec2, p=r) for r in r_values]

plt.plot(r_values, distances, marker='o', linestyle='--', color='green')
plt.title("Minkowski Distance between Two Feature Vectors (r=1 to 10)")
plt.xlabel("r (order of Minkowski distance)")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

#%%
from sklearn.preprocessing import LabelEncoder

# Encode string labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['fluency'])

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

X = np.vstack(df['embedding'].values)  # embeddings
y = y_encoded  # encoded labels

# A4. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A5. Train kNN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# A6. Test accuracy
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)

# A7. Prediction
predictions = knn.predict(X_test)

# A8. Varying k from 1 to 11
acc_list = []
for k in range(1, 12):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    acc_list.append(acc)

plt.plot(range(1, 12), acc_list, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k')
plt.show()

# A9. Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", conf_matrix)
target_names = [str(label) for label in label_encoder.classes_]

report = classification_report(
    y_test,
    predictions,
    labels=label_encoder.transform(label_encoder.classes_),
    target_names=target_names
)

print("Classification Report:\n", report)

