import pandas as pd 
import numpy as np 

# Load the dataset
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\Purchase_Data.csv"
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