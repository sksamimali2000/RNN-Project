# Time Series Forecasting of Airline Passengers using RNN

## Overview
This project demonstrates time series forecasting using a Recurrent Neural Network (RNN) with Keras. The dataset contains monthly totals of international airline passengers.

## Dataset
- The dataset used: `international-airline-passengers.csv`
- Target: Forecast the number of airline passengers based on previous time steps.

## Data Preparation
- Only the passenger count column is used.
- Data is scaled to the range [0, 1] using `MinMaxScaler`.
- Dataset split: 
    - 67% for training
    - 33% for testing

### Sequence Creation
A sliding window technique is used to create sequences of data:
```python
def create_dataset(data, k):
    dataX, dataY = [], []
    for i in range(data.shape[0] - k):
        x = data[i:i + k, 0]
        y = data[i + k, 0]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)
```



look_back = 12 (past 12 months used to predict next month).

Data reshaped to fit RNN input: (samples, look_back, 1).

Model Architecture
```Python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

Training

Trained for 10 epochs with batch size of 1.

Predictions

Model predicts on both training and testing datasets.

Predictions are inverse-transformed back to the original scale.

Visualization
```Python
Training Data vs Predictions
plt.plot(trainTrue, c='g', label='True Train Data')
plt.plot(trainPredict, c='b', label='Predicted Train Data')
plt.legend()
plt.show()
```

Combined True Data vs Combined Predictions
```Python
combinedPredicted = np.concatenate((trainPredict, testPredict))
combinedTrue = np.concatenate((trainTrue, testTrue))

plt.plot(combinedTrue, c='g', label='True Data')
plt.plot(combinedPredicted, c='b', label='Predicted Data')
plt.legend()
plt.show()
```

Conclusion

The model successfully predicts the airline passenger trends.

Further tuning and use of more advanced architectures (like LSTM or GRU) can improve performance.
