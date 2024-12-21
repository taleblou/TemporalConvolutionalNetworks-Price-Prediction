import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dropout, Dense, Conv1D, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
import os
from tqdm import tqdm
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score

# Check TensorFlow version and available devices
print(tf.__version__)
warnings.filterwarnings("ignore")
print("GPUs:", tf.config.list_physical_devices('GPU'))
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Using device: {device}")

name = "^GSPC"  # "GC=F", "EURUSD=X", "^GSPC"
file_path = f"TCN{name}.txt"

def text_write(text):
    print(text)
    # Write text to file
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(text + "\n")
    else:
        with open(file_path, 'w') as file:
            file.write(text + "\n")

# Load and preprocess data
def load_data(ticker):
    data = yf.download(ticker)
    return data

# Build the TCN model
def build_tnc_model(Xtrain,Y_train,Xval,Y_val):
    model = Sequential()

    # TCN layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(Xtrain, Y_train, epochs=20, batch_size=32, validation_data=(Xval, Y_val))

    return model

# Load BTC data and preprocess
data = load_data(name)
data.to_csv(f"{name}.csv")
data = data[["Close", "Open", "High", "Low"]]
data = data[-1000:]

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

# Shift the target variables for prediction
data["y_Close"] = data['Close'].shift(-1)
data["y_Open"] = data['Open'].shift(-1)
data["y_High"] = data['High'].shift(-1)
data["y_Low"] = data['Low'].shift(-1)
data.dropna(inplace=True)

X = data[["Close", "Open", "High", "Low"]]
Y = data[["y_Close", "y_Open", "y_High", "y_Low"]]

# Add columns for predicted values
for col in ['Low', 'High', 'Open', 'Close']:
    data[f"p_{col}"] = np.nan
    data[f"o_p_{col}"] = np.nan
    data[f"o_y_{col}"] = np.nan

# Box size for sliding window
box = 200
for i in tqdm(range(box, len(X))):
    X_train = X[i-box:i]
    Y_train = Y[i-box:i]
    X_val = X[i-box-50:i-box]
    Y_val = Y[i-box-50:i-box]
    X_test = X[i:i+1]
    Y_test = Y[i:i+1]

    for c in ['Open', 'High', 'Low', 'Close']:
        # Reshape the input for Conv1D (samples, timesteps, features)
        Xtrain = np.expand_dims(X_train, axis=-1)
        Xval = np.expand_dims(X_val, axis=-1)
        Xtest = np.expand_dims(X_test, axis=-1)

        with tf.device(device):
            model = build_tnc_model(Xtrain, Y_train["y_" + c], Xval, Y_val["y_" + c])

        # Predict
        predictions = model.predict(Xtest, verbose=0)
        data.loc[data.index[i], f'p_{c}'] = predictions[0][0]

        # Inverse transform predictions
        predictions = np.tile(predictions, 4).reshape(1, 4)
        predictions = scaler.inverse_transform(predictions)
        data.loc[data.index[i], f'o_p_{c}'] = predictions[0][0]

        # Inverse transform actual values
        target = scaler.inverse_transform(Y_test)
        data.loc[data.index[i], f'o_y_{c}'] = target[0][0]

# Calculate accuracy and plot results
df = data[['y_Open', 'p_Open', 'y_Close', 'p_Close', 'y_High', 'p_High', 'y_Low', 'p_Low', 'o_y_Open', 'o_p_Open', 'o_y_Close', 'o_p_Close', 'o_y_High', 'o_p_High', 'o_y_Low', 'o_p_Low']]
df.dropna(inplace=True)
df.to_csv(f"Predict_{name}.csv")

# Plot and calculate metrics
for c in ['Open', 'High', 'Low', 'Close']:
    mse = mean_squared_error(df['y_' + c], df['p_' + c])
    mae = mean_absolute_error(df['y_' + c], df['p_' + c])
    r2 = r2_score(df['y_' + c], df['p_' + c])
    medae = median_absolute_error(df['y_' + c], df['p_' + c])
    evs = explained_variance_score(df['y_' + c], df['p_' + c])

    text_write(f"Mean Squared Error({c}): {mse}")
    text_write(f"Mean Absolute Error({c}): {mae}")
    text_write(f"R-squared({c}): {r2}")
    text_write(f"Median Absolute Error({c}): {medae}")
    text_write(f"Explained Variance Score({c}): {evs}")

    fig, axes = plt.subplots()
    plt.plot(df.index, df['o_y_' + c], label='Actual ' + c + ' Price', color='blue')
    plt.plot(df['o_p_' + c], label='Predicted ' + c + ' Price', color='green')
    plt.title(f'{c} Price Prediction')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"TCN_{c}_{name}.png")
    plt.close()


