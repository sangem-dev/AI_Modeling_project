# Task 1: Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Check for missing values
print(df.isnull().sum())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df['target'], test_size=0.2, random_state=42
)

# Output shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
