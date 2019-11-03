import json

import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.layers.wrappers import TimeDistributed
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MaxAbsScaler
import argparse


def main(params):
    with open(params['file']) as f:
        for video in f:

            # preparing data

            dataframe = pd.read_csv(params['data'] + 'hawkes_' + video.strip() + '.csv', engine='python')
            data_x = dataframe['numShares'].values
            data_y = dataframe['numViews'].values
            data_x = data_x.astype('float32')
            data_y = data_y.astype('float32')

            scalerX = MaxAbsScaler()
            data_x = scalerX.fit_transform(data_x.reshape(-1, 1))
            scalerY = MaxAbsScaler()
            data_y = scalerY.fit_transform(data_y.reshape(-1, 1))

            # split into train and test sets
            train_index = 90
            test_index = 120

            x_train, y_train = data_x[:train_index], data_y[:train_index]
            X_test, y_test = data_x[train_index:test_index], data_y[train_index:test_index]

            # reshape input to be [samples, time steps, features]
            x_train = np.reshape(x_train, (1, x_train.shape[0], 1))
            X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
            y_train = np.reshape(y_train, (1, y_train.shape[0], 1))
            y_test = np.reshape(y_test, (1, y_test.shape[0], 1))

            print('x_train shape:', x_train.shape)
            print('X_test shape:', X_test.shape)
            print('y_train shape:', y_train.shape)
            print('y_test shape:', y_test.shape)

            hiddenStateSize = 10
            hiddenLayerSize = 10

            # train or predict

            if params['train']:
                print("Building training model for {}".format(video))
                model = Sequential()
                # The output of the LSTM layer are the hidden states of the LSTM for every time step.
                model.add(LSTM(hiddenStateSize, return_sequences=True, input_shape=(90, 1)))
                model.add(TimeDistributed(Dense(hiddenLayerSize)))
                model.add(TimeDistributed(Activation('relu')))
                model.add(TimeDistributed(Dense(1)))  # Add another dense layer with the desired output size.
                model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))
                print("Training model for {}".format(video))
                results = model.fit(x_train, y_train, batch_size=1, epochs=5000, verbose=0)
                # saving progress in a log file
                pd.DataFrame(results.history).to_csv('../logs/' + video.strip() + '.csv')
                # serialize weights to HDF5
                print("Saving trained model for {}".format(video))
                model.save_weights(params['model'] + video.strip() + '.h5')
            elif params['predict']:
                unit_response = {}
                # Two differences here.
                # 1. The inference model only takes one sample in the batch, and it always has sequence length 1.
                # 2. The inference model is stateful, meaning it inputs the output hidden state ("its history state")
                #    to the next batch input.
                inference_model.add(LSTM(hiddenStateSize, batch_input_shape=(1, 1, 1), stateful=True))
                # Since the above LSTM does not output sequences, we don't need TimeDistributed anymore.
                inference_model.add(Dense(hiddenLayerSize))
                inference_model.add(Activation('relu'))
                inference_model.add(Dense(1))
                # inference_model.add(Activation('softmax'))
                # Copy the weights of the trained network. Both should have the same exact number of parameters (why?).
                inference_model.load_weights(params['model'] + video.strip() + '.h5')

                trainPredict = np.zeros(90)

                for i in range(90):
                    trainPredict[i] = \
                    np.reshape(inference_model.predict(np.reshape(x_train[0, i, 0], (1, 1, 1))), (1))[0]


    print("Run Completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Predict volume RNN for videos provided.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("--file", type=str, default="video_ids.txt", help="file with video ids")
    parser.add_argument("--data", type=str, default="../data/", help="root folder for data")
    parser.add_argument("--model", type=str, default="../models/", help="root folder for models")
    parser.add_argument("--plot_folder", type=str, help="root folder for plots")

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed parameters:', json.dumps(params, indent=2))
    main(params)
