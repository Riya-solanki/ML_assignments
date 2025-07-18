import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\thyroid0387_UCI.csv"
)

# Step 2: Fill missing values (for simplicity, use 0)
data.fillna(0, inplace=True)

# Step 3: Convert 't'/'f' to 1/0 in the entire dataframe
data.replace({'t': 1, 'f': 0}, inplace=True)

# Step 4: Convert all object columns to numerical using LabelEncoder
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# Step 5: Get feature vectors for first two observations
v1 = data.iloc[0].values.reshape(1, -1)
v2 = data.iloc[1].values.reshape(1, -1)

# Step 6: Calculate Cosine Similarity
cos_sim = cosine_similarity(v1, v2)[0][0]

# Step 7: Print Result
print("Cosine Similarity between first two observations:", round(cos_sim, 4))

# Step 8: Get first 20 observations
first_20 = data.iloc[:20]

# Step 9: Calculate cosine similarity matrix
cos_matrix = cosine_similarity(first_20)

# Step 10: Plot heatmap for cosine similarity
plt.figure(figsize=(10, 8))
sns.heatmap(cos_matrix, annot=False, cmap='magma')
plt.title("Cosine Similarity (First 20 Observations)")
plt.show()
