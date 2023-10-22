# Google Stock Price Prediction 📈

Using Machine Learning and Deep Learning algorithms like Linear Regression (LR), Neural Networks (NN), and Recurrent Neural Networks (RNN) to predict Google's stock prices. 

## 📊 Algorithms:
1. **Linear Regression (LR)**
2. **Neural Networks (NN)**
3. **Recurrent Neural Networks (RNN)** - Demonstrated in the provided code.

## 📝 Code Highlights:
- **Data Preprocessing**: Uses `pandas` for data manipulation and `MinMaxScaler` from `sklearn` for normalization.
- **RNN Model Architecture**: Built using Keras with the following layers:
    1. LSTM layers with dropout for regularization.
    2. Dense output layer.
- **Visualization**: Displays real vs. predicted Google stock prices for 2017 using `matplotlib`.

## 🛠️ Libraries Used:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/stable/)

## 📁 Dataset:
- **Training Data**: 'Google_Stock_Price_Train.csv'
- **Testing Data**: 'Google_Stock_Price_Test.csv'

## 🚀 How to Run:
1. Ensure you have the above-mentioned libraries installed.
2. Load the training and testing datasets.
3. Run the Python script. The script will preprocess the data, train the RNN model, and then visualize the real and predicted Google stock prices for 2017.

## 📉 Results:
The visual output will showcase the actual Google stock price in red and the predicted price in blue. The title indicates the prediction for the year 2017.

> **Note**: The accuracy metric used in the example is not suitable for regression tasks. A metric like Mean Squared Error (MSE) would be more appropriate.

## 🔗 References:
- [LSTM Networks](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Keras Documentation](https://keras.io/)
- [Google Stock Data (example source)](https://www.google.com/finance) (Replace with your actual source if different)
  
## Author
**Fadi Helal**

## 📜 License:
This project is open-source under the BSD 3-Clause License.


