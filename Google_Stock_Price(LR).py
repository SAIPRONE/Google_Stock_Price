# -*- coding: utf-8 -*-
"""
@author: Fadi Helal
"""
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and preprocess the data
df = pd.read_csv("Google_Stock_Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df.index[::-1] # Reversing index as Time feature
df['CloseFuture'] = df['Close'].shift(-60) # Predicting 60 days into the future

# Split the dataset into train and test sets
train_df = df[185:].dropna()
test_df = df[:185]

# Model feature and target preparation
X_train = train_df['Time'].values.reshape(-1, 1)
y_train = train_df['CloseFuture'].values
X_test = test_df['Time'].values.reshape(-1, 1)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
train_df['Prediction'] = model.predict(X_train)
test_df['Prediction'] = model.predict(X_test)

# Plot the results
plt.scatter(df['Date'], df['Close'], color='black', label='Actual')
plt.plot(train_df['Date'] + pd.DateOffset(days=60), train_df['Prediction'], color = 'red', label='Predicted (Train)')
plt.plot(test_df['Date'] + pd.DateOffset(days=60), test_df['Prediction'], color = 'green', label='Predicted (Test)')
plt.legend()
plt.show()

# Evaluate the model
mae_train = mean_absolute_error(train_df['CloseFuture'], train_df['Prediction'])
mae_test = mean_absolute_error(test_df['CloseFuture'].dropna(), test_df['Prediction'].dropna())

print(f"Mean Absolute Error in Train data: {mae_train:.2f}")
print(f"Mean Absolute Error in Test data: {mae_test:.2f}")
