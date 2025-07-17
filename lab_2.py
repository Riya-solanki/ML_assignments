import pandas as pd 
import numpy as np 

# Load the dataset
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\Lab Session Data(Purchase data).csv"
)

# Get number of columns and determine A (quantities) and C (total cost)
num_cols = data.shape[1]

# Segregate A and C
A = data.iloc[:, :-1].values  # All columns except the last one
C = data.iloc[:, -1].values.reshape(-1, 1)  # Last column as a column vector

# Get dimension and number of vectors
dimen = A.shape[1]
numVect = A.shape[0]

# Convert A to numeric (handle any non-numeric entries)
A_numeric = pd.DataFrame(A).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Compute rank
rank = np.linalg.matrix_rank(A_numeric)
print(f"Rank of matrix A: {rank}")

# Compute pseudo-inverse of A
A_pinv = np.linalg.pinv(A_numeric)

# Solve for product costs (X)
X = A_pinv @ C

# Output estimated product costs
print("Estimated cost of each product:")
for i, cost in enumerate(X):
    print(f"Product {i+1}: ₹{round(float(cost), 2)}")


#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\Lab Session Data(Purchase data).csv"
)

# Step 2: Clean data
# Keep only the important columns
useful_cols = ['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']
data = data[useful_cols]

# Drop rows where Payment (Rs) is missing or not a number
data = data[pd.to_numeric(data['Payment (Rs)'], errors='coerce').notnull()]
data['Payment (Rs)'] = data['Payment (Rs)'].astype(float)

# Step 3: Add Label based on Payment
data['Label'] = data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Step 4: Feature selection
X = data.drop(columns=['Customer', 'Payment (Rs)', 'Label'])  # Features
y = data['Label']  # Target

# Encode categorical target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 6: Train classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Optional: Show predictions with customer names
data['Prediction'] = le.inverse_transform(model.predict(X))
print(data[['Customer', 'Payment (Rs)', 'Label', 'Prediction']])


#%%
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\Lab Session Data(IRCTC Stock Price).csv"
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


# 2. Sample mean of prices on Wednesdays
wednesday_prices = pd.to_numeric(data[data['Day'] == 'Wednesday'].iloc[:, 3], errors='coerce').dropna()
april_prices = pd.to_numeric(data[data['Date'].dt.month == 4].iloc[:, 3], errors='coerce').dropna()

print(f"\nSample mean (Wednesdays): {mean_wed:.2f}")
print("Observation:", "Higher" if mean_wed > mean_price else "Lower", "than population mean")

# 3. Sample mean of prices in April
april_prices = pd.to_numeric(data[data['Date'].dt.month == 4].iloc[:, 3], errors='coerce').dropna()
mean_apr = statistics.mean(april_prices)
print(f"\nSample mean (April): {mean_apr:.2f}")
print("Observation:", "Higher" if mean_apr > mean_price else "Lower", "than population mean")

# 4. Probability of making a loss (Chg% < 0)
chg_list = pd.to_numeric(data.iloc[:, 8], errors='coerce').dropna()
loss_days = list(filter(lambda x: x < 0, chg_list))
prob_loss = len(loss_days) / len(chg_list)
print(f"\nProbability of making a loss: {prob_loss:.2f}")

# 5. Probability of making profit on Wednesday
data['Chg%'] = pd.to_numeric(data.iloc[:, 8], errors='coerce')  # Create a proper Chg% column
profit_wed = data[(data['Day'] == 'Wednesday') & (data['Chg%'] > 0)]
total_wed = data[data['Day'] == 'Wednesday']
prob_profit_wed = len(profit_wed) / len(total_wed)
print(f"\nProbability of making profit on Wednesday: {prob_profit_wed:.2f}")

# 6. Conditional Probability (Same as #5)
print(f"\nConditional Probability P(Profit | Wednesday): {prob_profit_wed:.2f}")

# 7. Scatter plot: Chg% vs Day of the week
plt.figure(figsize=(10, 6))
sns.stripplot(x='Day', y='Chg%', data=data, jitter=True)
plt.title("Change % vs Day of the Week")
plt.ylabel("Chg%")
plt.xlabel("Day of the Week")
plt.grid(True)
plt.show()


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the data
file_path = "Lab Session Data.xlsx"
data = pd.read_csv(file_path, sheet_name="thyroid0387_UCI")

# Display the first few rows
print("First 5 rows of the dataset:\n", data.head())

# ----------------------------
# 1. Data Types and Attributes
# ----------------------------
print("\nData Types of Each Column:\n", data.dtypes)

# ----------------------------
# 2. Missing Value Check
# ----------------------------
print("\nMissing Values Count:\n", data.isnull().sum())

# ----------------------------
# 3. Identify Categorical & Numerical Columns
# ----------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

print("\nCategorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))

# ----------------------------
# 4. Encoding Categorical Columns
# ----------------------------
# Label Encoding for Ordinal (example shown)
label_enc = LabelEncoder()
data_encoded = data.copy()

for col in categorical_cols:
    try:
        data_encoded[col] = label_enc.fit_transform(data_encoded[col].astype(str))
    except:
        print(f"Could not encode {col}")

print("\nAfter Label Encoding:\n", data_encoded.head())

# ----------------------------
# 5. Data Range of Numeric Variables
# ----------------------------
print("\nRange of Numeric Variables:")
for col in numerical_cols:
    print(f"{col}: Min = {data[col].min()}, Max = {data[col].max()}")

# ----------------------------
# 6. Mean and Variance of Numeric Variables
# ----------------------------
print("\nMean and Standard Deviation of Numeric Columns:")
for col in numerical_cols:
    print(f"{col}: Mean = {data[col].mean():.2f}, Std Dev = {data[col].std():.2f}")

# ----------------------------
# 7. Outlier Detection using Boxplot
# ----------------------------
for col in numerical_cols:
    plt.figure()
    sns.boxplot(x=data[col])
    plt.title(f"Outlier Detection - {col}")
    plt.show()
