import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\thyroid0387_UCI.csv"
)


# 1. Data Types and Attributes
print("\nData Types of Each Column:\n", data.dtypes)


# 2. Missing Value Check
print("\nMissing Values Count:\n", data.isnull().sum())

# 3. Identify Categorical & Numerical Columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

print("\nCategorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))

# 4. Encoding Categorical Columns
# Label Encoding for Ordinal (example shown)
label_enc = LabelEncoder()
data_encoded = data.copy()

for col in categorical_cols:
    try:
        data_encoded[col] = label_enc.fit_transform(data_encoded[col].astype(str))
    except:
        print(f"Could not encode {col}")

print("\nAfter Label Encoding:\n", data_encoded.head())

# 5. Data Range of Numeric Variables
print("\nRange of Numeric Variables:")
for col in numerical_cols:
    print(f"{col}: Min = {data[col].min()}, Max = {data[col].max()}")

# 6. Mean and Variance of Numeric Variables
print("\nMean and Standard Deviation of Numeric Columns:")
for col in numerical_cols:
    print(f"{col}: Mean = {data[col].mean():.2f}, Std Dev = {data[col].std():.2f}")

# 7. Outlier Detection using Boxplot
for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=data[col])
    plt.title(f"Outlier Detection - {col}")
    plt.show()
