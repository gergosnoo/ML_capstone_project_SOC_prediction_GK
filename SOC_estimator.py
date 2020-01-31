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
    def __init__(self, data):
        self.x_train = data[0]
        self.y_train = data[1]
        self.x_test = data[2]
        self.y_test = data[3]

    def run(self):
        # Building the model
        model = Sequential()
        model.add(Dense(18, input_dim=len(self.x_train[0])))
        # (Dense(50, activation='relu', input_shape=self.x_train[0].shape))
        # model.add(BatchNormalization())
        model.add(Dense(12, activation='relu'))
        # model.add(Dense(120, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        # model.add(Dropout(.1))
        # model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))

        # Optimizer
        adam = optimizers.Adam(lr=0.0001)
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95)
        rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9)

        # Compiling the model
        model.compile(loss='mean_absolute_error', optimizer=adadelta,
                      metrics=['binary_accuracy', 'categorical_accuracy'])

        # [['accuracy'], ['accuracy', 'mse']]
        model.summary()

        # Training the model
        model.fit(self.x_train, self.y_train, epochs=300, validation_data=(self.x_test, self.y_test),
                  batch_size=500, verbose=0)

        return model


def scale_between_0_and_1(df, label):
    if min(df[label]) >= 0:
        return df[label] / max(df[label])
    else:
        return (df[label] + abs(min(df[label]))) / (max(df[label]) + abs(min(df[label])))


def scale_between_minus1_and_1(df, label):
    if abs(min(df[label])) <= max(df[label]):
        df[label] /= max(df[label])
    else:
        df[label] /= abs(min(df[label]))
    return df[label]


def normalize_data(df):
    # creating a list of dataframe columns
    columns = list(df)
    for i in columns:
        if i == 'I_m':
            df[i] = scale_between_minus1_and_1(df, i)
        elif i == 'mode' or i == 'B_E' or i == 'date':
            pass
        else:
            df[i] = scale_between_0_and_1(df, i)
    return df


def data_prep(data1, data2):
    # Prepare usable input parameters
    data1['I_m'] = round(data1['mode'] * data1['B_E'] * data1['I_m'], 3)
    data2['I_m'] = round(data2['mode'] * data2['B_E'] * data2['I_m'], 3)
    y_d1 = scale_between_0_and_1(data1, 'SoC')
    y_d2 = scale_between_0_and_1(data2, 'SoC')
    x = []
    y = [y_d1, y_d2]
    for data in [data1, data2]:
        # sample_id, actual_time, mode, B_E, I_m, U_b, T_1, T_2, T_3, delta_time, runtime, dV, dV2, dV3, C
        x_15p = data.drop(['date', 'SoC'], axis=1)
        x_15p = normalize_data(x_15p)

        # I_m, U_b, T_2, delta_time, runtime, dV, dV2, dV3, C
        x_9p = data.drop(['sample_id', 'date', 'actual_time', 'mode', 'B_E', 'T_1', 'T_3', 'SoC'], axis=1)
        x_9p = normalize_data(x_9p)

        # I_m, U_b, dV, dV2, dV3, C
        x_6p = data.drop(['sample_id', 'date', 'actual_time', 'mode', 'B_E',
                          'T_1', 'T_2', 'T_3', 'delta_time', 'runtime', 'SoC'], axis=1)
        x_6p = normalize_data(x_6p)

        # I_m, U_b, delta_time, runtime, dV
        x_5p_1 = data.drop(['sample_id', 'date', 'actual_time', 'mode', 'B_E', 'T_1', 'T_2', 'T_3', 'dV2', 'dV3', 'C',
                            'SoC'], axis=1)
        x_5p_1 = normalize_data(x_5p_1)

        # I_m, U_b, dV, dV2, dV3
        x_5p_2 = data.drop(['sample_id', 'date', 'actual_time', 'mode', 'B_E',
                            'T_1', 'T_2', 'T_3', 'delta_time', 'runtime', 'C', 'SoC'], axis=1)
        x_5p_2 = normalize_data(x_5p_2)

        # I_m, U_b, delta_time
        x_3p = data.drop(['sample_id', 'runtime', 'dV', 'date', 'actual_time', 'mode', 'B_E', 'T_1', 'T_2', 'T_3',
                          'dV2', 'dV3', 'C', 'SoC'], axis=1)
        x_3p = normalize_data(x_3p)

        # I_m, U_b, dV, dV2, dV3, C divided into 3 column groups based on charging mode of the battery
        # x_load has 18 parameters
        # Divide measurements based on charging load profile
        x_load = pd.DataFrame()

        x_load['ch1'] = x_6p['U_b']
        x_load['ch2'] = x_6p['dV']
        x_load['ch3'] = x_6p['dV2']
        x_load['ch4'] = x_6p['dV3']
        x_load['ch5'] = x_6p['C']
        x_load['ch6'] = x_6p['I_m']

        x_load['idle1'] = x_6p['U_b']
        x_load['idle2'] = x_6p['dV']
        x_load['idle3'] = x_6p['dV2']
        x_load['idle4'] = x_6p['dV3']
        x_load['idle5'] = x_6p['C']
        x_load['idle6'] = x_6p['I_m']

        x_load['dsch1'] = x_6p['U_b']
        x_load['dsch2'] = x_6p['dV']
        x_load['dsch3'] = x_6p['dV2']
        x_load['dsch4'] = x_6p['dV3']
        x_load['dsch5'] = x_6p['C']
        x_load['dsch6'] = x_6p['I_m']

        # Clear-up
        # 1st third of the inputs to represent charging 'ch'
        # 2nd third of the inputs to represent Standby or Bypass 'idle'
        # 3rd third of the inputs to represent discharging 'dsch'
        x_load['ch1'].values[x_6p['I_m'] <= 0] = 0
        x_load['ch2'].values[x_6p['I_m'] <= 0] = 0
        x_load['ch3'].values[x_6p['I_m'] <= 0] = 0
        x_load['ch4'].values[x_6p['I_m'] <= 0] = 0
        x_load['ch5'].values[x_6p['I_m'] <= 0] = 0
        x_load['ch6'].values[x_6p['I_m'] <= 0] = 0
        x_load['idle1'].values[x_6p['I_m'] != 0] = 0
        x_load['idle2'].values[x_6p['I_m'] != 0] = 0
        x_load['idle3'].values[x_6p['I_m'] != 0] = 0
        x_load['idle4'].values[x_6p['I_m'] != 0] = 0
        x_load['idle5'].values[x_6p['I_m'] != 0] = 0
        x_load['idle6'].values[x_6p['I_m'] != 0] = 0
        x_load['dsch1'].values[x_6p['I_m'] >= 0] = 0
        x_load['dsch2'].values[x_6p['I_m'] >= 0] = 0
        x_load['dsch3'].values[x_6p['I_m'] >= 0] = 0
        x_load['dsch4'].values[x_6p['I_m'] >= 0] = 0
        x_load['dsch5'].values[x_6p['I_m'] >= 0] = 0
        x_load['dsch6'].values[x_6p['I_m'] >= 0] = 0

        # I_m, U_b, dV, dV2, dV3 divided into 3 column groups based on charging mode of the battery
        # x_load_2 has 15 parameters
        x_load_2 = x_load.drop(['ch5', 'idle5', 'dsch5'], axis=1)

        x.append([x_15p, x_9p, x_5p_1, x_5p_2, x_3p, x_6p, x_load, x_load_2])

    data_sets = []

    for i in range(len(x)):
        for j in range(len(x[i])):
            x_train, y_train = x[i][j].to_numpy(), y[i].to_numpy()
            x_test, y_test = x[1 - i][j].to_numpy(), y[1 - i].to_numpy()
            data_sets.append([x_train, y_train, x_test, y_test])

    return data_sets


def plot_run(error, reference, prediction, label):
    plt.subplots(figsize=(10, 8))
    #
    plt.subplot(2, 1, 1)
    plt.title('x = [' + label + ']')
    plt.plot(reference, label='Reference SoC')
    plt.plot(prediction, label='Prediction SoC')
    # plt.plot(data['runtime'], data['I_m'], label='I_m')
    plt.xlabel('sample_id')
    plt.ylabel('SoC')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Error')
    plt.plot(error, label='Error in SoC prediction')
    # plt.plot(data['runtime'], data['I_m'], label='I_m')
    plt.xlabel('sample_id')
    plt.ylabel('Error function')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Load the dataset
data_1 = pd.read_csv('data/Augmented_data/battery_data_2_dV-C-roundV.csv')
print("battery_data_1 has {} data points with {} variables each."
      .format(*data_1.shape))
data_2 = pd.read_csv('data/Augmented_data/battery_data_2_dV-C.csv')
print("battery_data_2 has {} data points with {} variables each."
      .format(*data_2.shape))


data_sets = data_prep(data_1, data_2)


errors = []
accuracies = []
for i in range(len(data_sets)):
    soc_nn = soc_nn(data_sets[i])
    model = soc_nn.run()

    x_test = data_sets[i][2]
    y_test = data_sets[i][3]

    pred = model.predict(x_test, batch_size=32, verbose=0)
    prediction = []
    for j in range(len(pred)):
        prediction.append(pred[j][0])

    err = np.subtract(y_test, prediction)
    error = np.absolute(err)
    err_mean = np.mean(error)
    accuracies.append(1 - err_mean)
    errors.append(err_mean)
    print('Error: ', err_mean)
    print('Accuracy: ', 1 - err_mean)
    plot_run(error, y_test, prediction,
             'sample_id, actual_time, mode, B_E, I_m, U_b, T_1, T_2, T_3, delta_time, runtime, dV, dV2, dV3, C')