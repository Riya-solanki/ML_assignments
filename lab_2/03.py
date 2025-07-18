import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\IRCTC_Stock.csv"
)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract day of the week
data['Day'] = data['Date'].dt.day_name()

# Convert the column to numeric, forcing errors to NaN, then drop NaNs
# --- 1. Mean and Variance of Price (Column D) ---
price_list = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()
mean_price = statistics.mean(price_list)
var_price = statistics.variance(price_list)

print(f"Mean of Price: {mean_price:.2f}")
print(f"Variance of Price: {var_price:.2f}")
