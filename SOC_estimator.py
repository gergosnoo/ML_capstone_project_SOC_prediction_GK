# Imports
import numpy as np
import keras
from keras.datasets import imdb
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, regularizers
from keras import optimizers
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import tensorflow as tf
import random

np.random.seed(42)
# Print the number of GPUs
print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))


class soc_nn:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def run(self):
        # Building the model
        model = Sequential()
        model.add(Dense(18, input_dim=len(self.x_train[0])))
        # (Dense(50, activation='relu', input_shape=self.x_train[0].shape))
        # model.add(BatchNormalization())
        model.add(Dense(12, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01)))
        # model.add(Dropout(.1))
        # model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))

        # Optimizer
        adam = optimizers.Adam(lr=0.01)

        # Compiling the model
        model.compile(loss='mean_squared_error', optimizer=adam,
                      metrics=['accuracy'])
        model.summary()

        # Training the model
        model.fit(self.x_train, self.y_train, epochs=500, batch_size=5000, verbose=0)

        # Evaluating the model on the training and testing set
        score = model.evaluate(self.x_train, self.y_train)
        print("\n Training Accuracy:", score[1])
        score = model.evaluate(self.x_test, self.y_test)
        print("\n Testing Accuracy:", score[1])

        # Running and evaluating the model
        hist = model.fit(self.x_train, self.y_train,
                         batch_size=64,
                         epochs=10,
                         validation_data=(self.x_test, self.y_test),
                         verbose=2)

        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print("Accuracy: ", score[1])


def scale_between_0_and_1(df, label):
   return (df[label] + abs(min(df[label]))) / (max(df[label]) + abs(min(df[label])))


# Load the dataset
data_1 = pd.read_csv('data/Augmented_data/battery_data_1_dV-C-roundV.csv')
print("battery_data_1 has {} data points with {} variables each."
      .format(*data_1.shape))
data_2 = pd.read_csv('data/Augmented_data/battery_data_2_dV-C.csv')
print("battery_data_2 has {} data points with {} variables each."
      .format(*data_2.shape))


# Prepare input parameters
y_1 = scale_between_0_and_1(data_1, 'SoC')
x_1 = data_1.drop(['sample_id', 'date', 'actual_time', 'mode', 'B_E', 'I_m',
                   'T_1', 'T_2', 'T_3', 'delta_time', 'runtime', 'SoC'], axis=1)

# x_1['T'] = round((data_1['T_1'] + data_1['T_2'] + data_1['T_3']) / 3, 1)
x_1['I_m'] = round(data_1['mode'] * data_1['B_E'] * data_1['I_m'], 3)
x_1['I'] = scale_between_0_and_1(x_1, 'I_m')
x_1['C'] = scale_between_0_and_1(data_1, 'C')
x_1['U_b'] = scale_between_0_and_1(data_1, 'U_b')
x_1['dV'] = scale_between_0_and_1(data_1, 'dV')
x_1['dV2'] = scale_between_0_and_1(data_1, 'dV2')
x_1['dV3'] = scale_between_0_and_1(data_1, 'dV3')

xt_1 = pd.DataFrame()
xt_1['ch1'] = x_1['U_b']
xt_1['ch2'] = x_1['dV']
xt_1['ch3'] = x_1['dV2']
xt_1['ch4'] = x_1['dV3']
xt_1['ch5'] = x_1['C']
xt_1['ch6'] = round(data_1['mode'] * data_1['B_E'] * data_1['I_m'], 3)
xt_1['idle1'] = x_1['U_b']
xt_1['idle2'] = x_1['dV']
xt_1['idle3'] = x_1['dV2']
xt_1['idle4'] = x_1['dV3']
xt_1['idle5'] = x_1['C']
xt_1['idle6'] = round(data_1['mode'] * data_1['B_E'] * data_1['I_m'], 3)
xt_1['dsch1'] = x_1['U_b']
xt_1['dsch2'] = x_1['dV']
xt_1['dsch3'] = x_1['dV2']
xt_1['dsch4'] = x_1['dV3']
xt_1['dsch5'] = x_1['C']
xt_1['dsch6'] = round(data_1['mode'] * data_1['B_E'] * data_1['I_m'], 3)

xt_1['ch1'].values[x_1['I_m'] <= 0] = 0
xt_1['ch2'].values[x_1['I_m'] <= 0] = 0
xt_1['ch3'].values[x_1['I_m'] <= 0] = 0
xt_1['ch4'].values[x_1['I_m'] <= 0] = 0
xt_1['ch5'].values[x_1['I_m'] <= 0] = 0
xt_1['ch6'].values[x_1['I_m'] <= 0] = 0
xt_1['idle1'].values[x_1['I_m'] != 0] = 0
xt_1['idle2'].values[x_1['I_m'] != 0] = 0
xt_1['idle3'].values[x_1['I_m'] != 0] = 0
xt_1['idle4'].values[x_1['I_m'] != 0] = 0
xt_1['idle5'].values[x_1['I_m'] != 0] = 0
xt_1['idle6'].values[x_1['I_m'] != 0] = 0
xt_1['dsch1'].values[x_1['I_m'] >= 0] = 0
xt_1['dsch2'].values[x_1['I_m'] >= 0] = 0
xt_1['dsch3'].values[x_1['I_m'] >= 0] = 0
xt_1['dsch4'].values[x_1['I_m'] >= 0] = 0
xt_1['dsch5'].values[x_1['I_m'] >= 0] = 0
xt_1['dsch6'].values[x_1['I_m'] >= 0] = 0


xt_1['ch6'] = xt_1['ch6'] / (max(xt_1['ch6']))
xt_1['dsch6'] = xt_1['dsch6'] / abs(min(xt_1['dsch6']))



print(xt_1, y_1)
x_1 = xt_1


# Create training and testing dataset
x_1, y_1 = x_1.to_numpy(), y_1.to_numpy()
train_size = int(len(x_1)/2)
test_size = len(x_1) - train_size

x_train = x_1[:train_size]
y_train = y_1[:train_size]
x_test = x_1[train_size:]
y_test = y_1[train_size:]

print("x_train has {} data points with {} variables each.".format(*x_train.shape))
print("x_test has {} data points with {} variables each.".format(*x_test.shape))

soc_nn = soc_nn(x_train, y_train, x_test, y_test)
soc_nn.run()