# -*- coding: utf-8 -*-
"""
@author: Fadi Helal
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Loading and preprocessing data
df = pd.read_csv("Google_Stock_Price.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df.index[::-1]
df['CloseFuture'] = df['Close'].shift(-30)

# Splitting the data into train and test datasets
df_train = df[185:].dropna()
df_test = df[:185]

# Scaling the 'Time' feature
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train[['Time']])
X_test = scaler.transform(df_test[['Time']])

# Extracting the target variable
y_train = df_train['CloseFuture'].values
y_test = df_test['CloseFuture'].values

# Building and training the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, activation="sigmoid", input_shape=(1,)),
    tf.keras.layers.Dense(20, activation="sigmoid"),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=120, batch_size=10)

# Making predictions
df_train['Prediction'] = model.predict(X_train)
df_test['Prediction'] = model.predict(X_test)

# Plotting the results
plt.scatter(df['Date'], df['Close'], color='black')
plt.plot(df_train['Date'], df_train['Prediction'], color='red')
plt.plot(df_test['Date'], df_test['Prediction'], color='green')
plt.show()

# Calculating and printing the mean absolute error
print(f"Mean absolute error in the test data: {mean_absolute_error(df_test['CloseFuture'], df_test['Prediction']):.2f}")
print(f"Mean absolute error in the training data: {mean_absolute_error(df_train['CloseFuture'], df_train['Prediction']):.2f}")
