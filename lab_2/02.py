import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load data
data = pd.read_csv(
    r"C:\Users\hp\OneDrive - Amrita vishwa vidyapeetham\Documents\Machine Learning\ML_assignments\lab_2\Purchase_Data.csv"
)

# Step 2: Clean data
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

