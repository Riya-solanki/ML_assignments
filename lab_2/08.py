import pandas as pd
import numpy as np
from scipy.stats import iqr

# Load the dataset
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\thyroid0387_UCI.csv"
)

# Function to check for outliers using IQR
def has_outliers(series):
    if not np.issubdtype(series.dtype, np.number):
        return False
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    return ((series < lower_bound) | (series > upper_bound)).any()

# Impute missing values appropriately
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'object':
            # Mode for categorical columns
            mode_val = data[col].mode()[0]
            data[col].fillna(mode_val, inplace=True)
        elif has_outliers(data[col]):
            # Median for numeric columns with outliers
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
        else:
            # Mean for numeric columns without outliers
            mean_val = data[col].mean()
            data[col].fillna(mean_val, inplace=True)

# Show updated null counts
print("Missing values after imputation:")
print(data.isnull().sum())
