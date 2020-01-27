# Imports
import numpy as np
import keras
from keras.datasets import imdb
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

# Load the Boston housing dataset
data = pd.read_csv('data/battery_data_1.csv')
print("battery_data_1 has {} data points with {} variables each.".format(*data.shape))

y = data['SoC']
x = data.drop(['SoC', 'sample_id', 'date', 'actual_time', 'runtime', 'T_1', 'T_2', 'T_3'], axis=1)
x['T'] = round((data['T_1'] + data['T_2'] + data['T_3']) / 3, 1)

print(x, y)

x, y = x.to_numpy(), y.to_numpy()
train_size = int(len(x)/2)
test_size = len(x) - train_size

x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

print("x_train has {} data points with {} variables each.".format(*x_train.shape))
print("x_test has {} data points with {} variables each.".format(*x_test.shape))

# TODO: Build the model architecture

# TODO: Compile the model using a loss function and an optimizer.

# Building the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=x_train[0].shape))
model.add(Dropout(.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(560, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# Training the model
model.fit(x_train, y_train, epochs=100, batch_size=10000, verbose=0)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(x_test, y_test)
print("\n Testing Accuracy:", score[1])

# TODO: Run the model. Feel free to experiment with different batch sizes and
#  number of epochs.
# Running and evaluating the model
hist = model.fit(x_train, y_train,
                 batch_size=32,
                 epochs=10,
                 validation_data=(x_test, y_test),
                 verbose=2)


score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
