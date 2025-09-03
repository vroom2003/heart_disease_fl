# load_data.py
import pandas as pd
import numpy as np
import os

print("🧹 Cleaning Heart Disease Dataset...")

# Create data folder if not exists
os.makedirs('data', exist_ok=True)

# Download and load raw data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

df = pd.read_csv(url, names=columns, na_values='?')

# Fix target: 1 = has disease, 0 = no disease
df['target'] = (df['target'] > 0).astype(int)

# Fill missing values
df['ca'].fillna(0, inplace=True)
df['thal'].fillna(3, inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['cp', 'slope', 'thal'], drop_first=True)

# Save cleaned data
df.to_csv('data/heart_disease_clean.csv', index=False)
print("✅ Cleaned data saved to data/heart_disease_clean.csv")
print(f"📊 Shape: {df.shape} | Heart Disease: {df['target'].sum()}/{len(df)}")