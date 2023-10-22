# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:14:43 2022

@author: Saibrone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from sklearn import preprocessing


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = df.index[::-1]  # Reverse the index, as the name suggests.
    df['CloseFuture'] = df['Close'].shift(-30)  # Shift to the past, not the future.

    df_train = df[:-185]
    df_test = df[-185:]

    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(df_train[['Time', 'Close']])
    X_test_scaled = scaler.transform(df_test[['Time', 'Close']])

    return df_train, df_test, X_train_scaled, df_train['CloseFuture'].values, X_test_scaled

def build_and_train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',  
                  metrics=['mae'])

    model.fit(X_train, y_train, epochs=100, batch_size=10)

    return model

def evaluate_model(model, X_train, df_train, X_test, df_test):
    df_train['PredictedClose'] = model.predict(X_train)
    df_test['PredictedClose'] = model.predict(X_test)

    plt.scatter(df_train['Date'], df_train['Close'], color='black')
    plt.plot(df_train['Date'], df_train['PredictedClose'], color='red')
    plt.scatter(df_test['Date'], df_test['Close'], color='black')
    plt.plot(df_test['Date'], df_test['PredictedClose'], color='green')
    plt.show()

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    print("Mean absolute error in the training data: %.2f" %
          mean_absolute_error(df_train['CloseFuture'], df_train['PredictedClose']))
    print("Mean absolute error in the test data: %.2f" %
          mean_absolute_error(df_test['CloseFuture'], df_test['PredictedClose']))

if __name__ == "__main__":
    df_train, df_test, X_train, y_train, X_test = load_and_preprocess_data("Google_Stock_Price.csv")
    model = build_and_train_model(X_train, y_train)
    evaluate_model(model, X_train, df_train, X_test, df_test)
