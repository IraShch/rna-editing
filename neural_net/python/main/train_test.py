import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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


def fill_identical_column_randomly(x, A_i, C_i, G_i, T_i, coverage_i, nucleotide):
    if nucleotide == 'A':
        return x[coverage_i] * (random.random() < 0.25)
    if nucleotide == 'C':
        return x[coverage_i] * (x[A_i] == 0) * (random.random() < 1 / float(3))
    if nucleotide == 'G':
        return x[coverage_i] * (x[A_i] + x[C_i] == 0) * (random.random() < 0.5)
    if nucleotide == 'T':
        return x[coverage_i] * (x[A_i] + x[C_i] + x[G_i] == 0)

def add_identical(X, y, k):
    print "Identical positions generation"
    n_additional = int(X.shape[0] * k)
    X_add = X.sample(n=n_additional, replace=True)

    A_i = X_add.columns.get_loc('A')
    C_i = X_add.columns.get_loc('C')
    G_i = X_add.columns.get_loc('G')
    T_i = X_add.columns.get_loc('T')
    coverage_i = X_add.columns.get_loc('coverage')

    # order of nucleotides is very important here!! Do not change rows order!
    print 'A'
    X_add['A'] = X_add.apply(lambda x: fill_identical_column_randomly(x, A_i, C_i, G_i, T_i, coverage_i, 'A'), axis=1)
    print 'C'
    X_add['C'] = X_add.apply(lambda x: fill_identical_column_randomly(x, A_i, C_i, G_i, T_i, coverage_i, 'C'), axis=1)
    print 'G'
    X_add['G'] = X_add.apply(lambda x: fill_identical_column_randomly(x, A_i, C_i, G_i, T_i, coverage_i, 'G'), axis=1)
    print 'T'
    X_add['T'] = X_add.apply(lambda x: fill_identical_column_randomly(x, A_i, C_i, G_i, T_i, coverage_i, 'T'), axis=1)

    X = pd.concat([X, X_add], axis=0)

    # create y
    y_add = pd.DataFrame(X_add, columns=['A', 'C', 'G', 'T'])
    y = pd.concat([y, y_add], axis=0)

    return X, y


# prepares data for learning
def split_data(data_dir, data_name, percent_train, seed, include_coverage, train_on_identical, k=1):
    # load initial noise set
    if include_coverage:
        noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y_cov.tsv'.format(data_dir, data_name)
    else:
        noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X = pd.read_table(noise_X_file_name)
    y = pd.read_table(noise_y_file_name)

    # split into train/validate/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - percent_train, random_state=seed)

    if train_on_identical:
        X_train, y_train = add_identical(X_train, y_train, k)

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
def train_test(X_train, X_test, y_train, y_test, nodes_number,
               batch_size, nb_epoch, output_dir, data_name, include_coverage, loss, optim):
    # define model structure
    model = Sequential()
    if include_coverage:
        model.add(Dense(nodes_number, input_dim=5, init='normal', activation='tanh'))
    else:
        model.add(Dense(nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.compile(loss=loss, optimizer=optim, metrics=['mean_squared_error', mean_residual_noise])
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
    parser.add_argument('-v', '--includeCoverage', help='include coverage column into X', action='store_true')
    parser.add_argument('-s', '--identicalPositions', help='include identical positions into training set',
                        action='store_true')
    parser.add_argument('-k', '--identicalPercent', help='number of identical positions added to trainig dataset: '
                                                         'k * nrow(X_train)', default=1)
    parser.add_argument('-l', '--loss', help='specify loss function to use: '
                                             'poisson (default) or mse', default='poisson')
    parser.add_argument('-t', '--optimizer', help='specify optimizer to use: '
                                                 'rmsprop (default) or adam', default='rmsprop')

    args = parser.parse_args()

    data_dir = args.dataDir
    if not data_dir.endswith('/'):
        data_dir += '/'
    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'
    data_name = args.dataName

    if args.includeCoverage:
        include_coverage = True
    else:
        include_coverage = False

    if args.identicalPositions:
        train_on_identical = True
        percent_identical = float(args.identicalPercent)
    else:
        train_on_identical = False
        percent_identical = 0

    loss = args.loss
    if loss not in ['poisson', 'mse']:
        raise ValueError('Loss function can be only poisson or mse!')

    opt = args.optimizer
    if opt not in ['rmsprop', 'adam']:
        raise ValueError('Optimizer can be only rmsprop or adam!')

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
    out_dir += '/train_test_{}nodes_{}epochs_{}coverage_{}identical_{}loss_{}opt'.format(nodes_number, nb_epoch,
                                                                            int(include_coverage),
                                                                            percent_identical,
                                                                            loss,
                                                                            opt)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    # write logs
    log_file = out_dir + 'log_test.txt'
    with open(log_file, 'w') as out:
        out.write('Data directory: {}\n'.format(data_dir))
        out.write('Percent of data to train model: {}\n'.format(percent_train))
        out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
        out.write('Number of epochs: {}\n'.format(nb_epoch))
        out.write('Batch size: {}\n'.format(batch_size))
        out.write('Loss function: {}\n'.format(loss))
        out.write('Optimizer: {}\n'.format(opt))
        out.write('Input coverage: {}\n'.format(include_coverage))
        out.write('Include identical positions into training set: {}\n'.format(train_on_identical))
        if train_on_identical:
            out.write('Identical rows to be added: {} * nrow(X_train)\n'.format(percent_identical))

    # fix random
    seed = 1214
    np.random.seed(seed)

    # split into sets
    X_train, X_test, y_train, y_test = split_data(data_dir, data_name, percent_train,
                                                  seed, include_coverage, train_on_identical, percent_identical)
    # run
    train_test(X_train, X_test, y_train, y_test, nodes_number,
               batch_size, nb_epoch, out_dir, data_name, include_coverage, loss, opt)


if __name__ == "__main__":
    main()
