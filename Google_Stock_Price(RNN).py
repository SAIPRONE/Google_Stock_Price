# -*- coding: utf-8 -*-
"""
@author: Fadi Helal
"""
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Data Preprocessing
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train['Open'].values.reshape(-1,1)

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = [training_set_scaled[i-60:i, 0] for i in range(60, 1258)]
y_train = [training_set_scaled[i, 0] for i in range(60, 1258)]
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Building the RNN
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Making the predictions and visualising the results
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test['Open'].values.reshape(-1,1)

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = np.array([inputs[i-60:i, 0] for i in range(60, 80)])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction for 2017')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

