import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

# calculates mean noise remained on "zero" positions
# approximation for numpy version
# to be Keras parameter
def mean_residual_noise(y_true, y_pred):
    a = 1
    k = 1000000
    return K.mean(K.sum((a / (k * K.pow(y_true, 2) + 1)) * y_pred, axis=1))


# prepares data for learning
def split_data(data_dir, data_name, percent_train, seed):
    # load initial noise set
    noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
    noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X = pd.read_table(noise_X_file_name)
    y = pd.read_table(noise_y_file_name)

    # split into train/validate/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - percent_train, random_state=seed)

    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test


# visualize learning history
def show_learning_history(history, data_name, output_dir):
    fname = output_dir + data_name
    # loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function ({})'.format(data_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname + '_loss.png')
    plt.clf()
    # MSE
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('MSE ({})'.format(data_name))
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname + '_mse.png')
    plt.clf()
    # residual noise for zero positions
    plt.plot(history.history['mean_residual_noise'])
    plt.plot(history.history['val_mean_residual_noise'])
    plt.title('Residual noise ({})'.format(data_name))
    plt.ylabel('noise')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(fname + '_mrn.png')
    plt.clf()


def test_model(model, X, y, model_name, output_dir):
    y_est = model.predict(X)
    squared_error = (y - y_est) ** 2
    residual_noise_sum = np.sum((y == 0) * y_est, axis=1)
    reference_position_noise = np.sum((y != 0) * (y - y_est), axis=1)
    result_table = np.concatenate((y, y_est, squared_error,
                                   np.column_stack((residual_noise_sum, reference_position_noise))), axis=1)
    output_file_name = output_dir + model_name + "_test_results.tsv"
    np.savetxt(output_file_name, result_table, fmt='%.3f', delimiter='\t', comments='',
               header='A\tC\tG\tT\tA_est\tC_est\tG_est\tT_est\tA_se\tC_se\tG_se\tT_se\tresidual_noise\treference_error')


# creates model, runs tests
def train_test(X_train, X_test, y_train, y_test, nodes_number, batch_size, nb_epoch, output_dir, data_name):
    # define model structure
    model = Sequential()
    model.add(Dense(nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.compile(loss='poisson', optimizer='rmsprop', metrics=['mean_squared_error', mean_residual_noise])
    # learn
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=0, validation_data=(X_test, y_test))

    # prepare directory for the results
    # create directory for history plots
    plots_dir = output_dir + '/history_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plots_dir += '/'

    # test
    show_learning_history(history, data_name, plots_dir)
    test_model(model, X_test, y_test, data_name, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
    parser.add_argument('-o', '--outputDir', help='directory to save the results in', required=True)
    parser.add_argument('-p', '--trainPercent', help='percent of data to use for training (0 < p < 1)', default=0.85)
    parser.add_argument('-n', '--nodesNumber', help='number of nodes in the hidden layer', default=50)
    parser.add_argument('-b', '--batchSize', help='batch size', default=4096)
    parser.add_argument('-e', '--epochNumber', help='number of epochs', default=400)

    args = parser.parse_args()

    data_dir = args.dataDir
    if not data_dir.endswith('/'):
        data_dir += '/'
    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'
    data_name = args.dataName

    percent_train = float(args.trainPercent)
    if percent_train >= 1 or percent_train <= 0:
        raise ValueError('Training data percent must be 0 < p < 1!')
    nodes_number = int(args.nodesNumber)
    batch_size = int(args.batchSize)
    nb_epoch = int(args.epochNumber)

    # prepare directory for the results
    # create directory with all results for this dataset
    out_dir += data_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create directory with train_test result
    out_dir += '/train_test_{}nodes_{}epochs'.format(nodes_number, nb_epoch)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    # write logs
    log_file = out_dir + 'log.txt'
    with open(log_file, 'w') as out:
        out.write('Data directory: {}\n'.format(data_dir))
        out.write('Percent of data to train model: {}\n'.format(percent_train))
        out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
        out.write('Number of epochs: {}\n'.format(nb_epoch))
        out.write('Batch size: {}\n'.format(batch_size))

    # fix random
    seed = 1214
    np.random.seed(seed)

    # split into sets
    X_train, X_test, y_train, y_test = split_data(data_dir, data_name, percent_train, seed)
    # run
    train_test(X_train, X_test, y_train, y_test, nodes_number, batch_size, nb_epoch, out_dir, data_name)


if __name__ == "__main__":
    main()
