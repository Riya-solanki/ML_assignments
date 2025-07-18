import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\thyroid0387_UCI.csv"
)
# Select only numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# See value ranges
print(data[numeric_cols].describe())
