# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)

# Loading the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)

print(x_train[0])
print('\n')
print(y_train[0])

# TODO: Build the model architecture

# TODO: Compile the model using a loss function and an optimizer.

# Building the model
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=x_train[0].shape))
model.add(Dropout(.2))
model.add(Dense(30, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='sigmoid'))

# Compiling the model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# Training the model
model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose=0)

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


