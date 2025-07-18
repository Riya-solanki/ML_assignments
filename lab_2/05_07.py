import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\thyroid0387_UCI.csv"
)

# Fill missing values with 0
data.fillna(0, inplace=True)

# Take only first 20 observations
subset = data.iloc[:20]

# Identify binary columns (with 't' and 'f')
binary_columns = []
for col in subset.columns:
    unique_vals = set(subset[col].astype(str).unique())
    if unique_vals.issubset({'t', 'f'}):
        binary_columns.append(col)

# Convert 't'/'f' to 1/0 in those columns
subset_binary = subset[binary_columns].replace({'t': 1, 'f': 0}).astype(int)

# Initialize matrices
n = len(subset_binary)
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))

# Calculate JC and SMC for each pair
for i in range(n):
    for j in range(n):
        v1 = subset_binary.iloc[i]
        v2 = subset_binary.iloc[j]
        
        f11 = np.sum((v1 == 1) & (v2 == 1))
        f10 = np.sum((v1 == 1) & (v2 == 0))
        f01 = np.sum((v1 == 0) & (v2 == 1))
        f00 = np.sum((v1 == 0) & (v2 == 0))

        jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
        smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) != 0 else 0

        jc_matrix[i, j] = jc
        smc_matrix[i, j] = smc

# Plot heatmaps
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(jc_matrix, annot=False, cmap='coolwarm')
plt.title("Jaccard Coefficient Heatmap")

plt.subplot(1, 2, 2)
sns.heatmap(smc_matrix, annot=False, cmap='viridis')
plt.title("Simple Matching Coefficient Heatmap")

plt.tight_layout()
plt.show()
