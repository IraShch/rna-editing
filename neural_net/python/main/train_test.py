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
from keras.regularizers import l2

# calculates mean noise remained on "zero" positions
# approximation for numpy version
# to be Keras parameter
def mean_residual_noise(y_true, y_pred):
    a = 1
    k = 1000000
    return K.mean(K.sum((a / (k * K.pow(y_true, 2) + 1)) * y_pred, axis=1))


# prepares data for learning
def split_data(data_dir, data_name, percent_train, seed, scaling_groups_number, add_noise):
    # load initial noise set
    noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
    noise_y_file_name = '{}{}_noise_y_cov.tsv'.format(data_dir, data_name)
    X = pd.read_table(noise_X_file_name)
    y = pd.read_table(noise_y_file_name)

    # scale coverage
    if scaling_groups_number == 1:
        med_coverage = X.median()['coverage']
        X['coverage'] /= med_coverage
    if scaling_groups_number == 2:
        X['group'] = (X['coverage'] > 1000).astype(int)
        penguins = X[X['group'] == 0].reset_index(drop=True)
        bears = X[X['group'] == 1].reset_index(drop=True)
        med_coverage = penguins.median()['coverage']
        penguins['coverage'] /= med_coverage
        med_coverage = bears.median()['coverage']
        bears['coverage'] /= med_coverage
        X = pd.concat([penguins, bears], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

    # create fractions
    coverage = X['A'] + X['C'] + X['G'] + X['T']
    X['A'] /= coverage
    X['C'] /= coverage
    X['G'] /= coverage
    X['T'] /= coverage

    if add_noise > 0:
        main_part = random.uniform(add_noise, 1)
        additional = (1 - main_part) / float(3)
        y['A'] = (y['A'] > 0) * (main_part - additional) + additional
        y['C'] = (y['C'] > 0) * (main_part - additional) + additional
        y['G'] = (y['G'] > 0) * (main_part - additional) + additional
        y['T'] = (y['T'] > 0) * (main_part - additional) + additional
    else:
        y['A'] = y['A'] > 0
        y['C'] = y['C'] > 0
        y['G'] = y['G'] > 0
        y['T'] = y['T'] > 0
        y = y.astype(int)

    # split into train/validate/test
    if percent_train == 1:
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
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
def train_test(X_train, X_test, y_train, y_test, nodes_number, batch_size, nb_epoch, output_dir, data_name, optim,
               scale_in_groups, activation, reg_const):
    # define model structure
    model = Sequential()
    if scale_in_groups == 2:
        model.add(Dense(nodes_number, input_dim=6, init='normal', activation='tanh', W_regularizer=l2(reg_const)))
    else:
        model.add(Dense(nodes_number, input_dim=5, init='normal', activation='tanh', W_regularizer=l2(reg_const)))
    model.add(Dense(4, init='normal', activation=activation, W_regularizer=l2(reg_const)))
    model.compile(loss='mse', optimizer=optim, metrics=[mean_residual_noise])
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
    parser.add_argument('-p', '--trainPercent', help='percent of data to use for training (0 < p <= 1)', default=0.85)
    parser.add_argument('-n', '--nodesNumber', help='number of nodes in the hidden layer', default=50)
    parser.add_argument('-b', '--batchSize', help='batch size', default=4096)
    parser.add_argument('-e', '--epochNumber', help='number of epochs', default=400)
    parser.add_argument('-t', '--optimizer', help='specify optimizer to use: '
                                                 'rmsprop (default) or adam', default='rmsprop')
    parser.add_argument('-a', '--activation', help='output layer activation function: relu (default) or softmax',
                        default='relu')
    parser.add_argument('-g', '--groupsToScale', help='In how many groups split dataset while scaling coverage (0, 1, 2)',
                        default=0)
    parser.add_argument('-s', '--addNoise', help='How much noise should be added to the y in training set', default=0)
    parser.add_argument('-l', '--lTwoConstant', help='l2 regularisation constant', default=0)

    args = parser.parse_args()

    data_dir = args.dataDir
    if not data_dir.endswith('/'):
        data_dir += '/'
    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'
    data_name = args.dataName

    opt = args.optimizer
    if opt not in ['rmsprop', 'adam']:
        raise ValueError('Optimizer can be only rmsprop or adam!')

    activation = args.activation
    if activation not in ['relu', 'softmax']:
        raise ValueError('Activation function can be only relu or softmax!')

    percent_train = float(args.trainPercent)
    if percent_train > 1 or percent_train <= 0:
        raise ValueError('Training data percent must be 0 < p <= 1!')
    nodes_number = int(args.nodesNumber)
    batch_size = int(args.batchSize)
    nb_epoch = int(args.epochNumber)

    scaling_groups_number = int(args.groupsToScale)
    if scaling_groups_number not in [0, 1, 2]:
        raise ValueError('Number of groups may be only 0, 1 or 2!')

    l2_reg_const = float(argparse.lTwoConstant)
    add_noise = float(args.addNoise)

    # prepare directory for the results
    # create directory with all results for this dataset
    out_dir += data_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create directory with train_test result
    additional_string = ''
    if scaling_groups_number > 0:
        additional_string += '_{}scaling'.format(scaling_groups_number)
    if activation == 'softmax':
        additional_string += '_softmax'
    if opt != 'rmsprop':
        additional_string += '_{}'.format(opt)
    if l2_reg_const:
        additional_string += '_l2{}'.format(l2_reg_const)
    if add_noise > 0:
        additional_string += '_noise{}'.format(add_noise)

    out_dir += '/train_test_{}nodes_{}epochs{}'.format(nodes_number, nb_epoch, additional_string)
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
        out.write('Loss function: MSE\n')
        out.write('Optimizer: {}\n'.format(opt))
        out.write('Scale coverage: {}\n'.format(scaling_groups_number > 0))
        if scaling_groups_number > 0:
            out.write('Split into two groups by coverage: {}\n'.format(scaling_groups_number == 2))
        out.write('Activation function: {}\n'.format(activation))
        out.write('L2 regularisation constant: {}\n'.format(l2_reg_const))

    # fix random
    seed = 1214
    np.random.seed(seed)

    # split into sets
    X_train, X_test, y_train, y_test = split_data(data_dir, data_name, percent_train, seed, scaling_groups_number,
                                                  add_noise)
    # run
    train_test(X_train, X_test, y_train, y_test, nodes_number, batch_size, nb_epoch, out_dir, data_name, opt,
               scaling_groups_number, activation, l2_reg_const)


if __name__ == "__main__":
    main()
