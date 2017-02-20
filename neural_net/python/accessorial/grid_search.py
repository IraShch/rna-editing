#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# a patch for Keras
# http://stackoverflow.com/questions/41796618/python-keras-cross-val-score-error
from keras.wrappers.scikit_learn import BaseWrapper
import copy


def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params


# prepares data for learning
def split_data(seed, percent_train, validate_test_ratio):
    # load initial noise set
    data_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/"
    noise_X_file_name = data_dir + "TNOR3_noise_X.tsv"
    noise_y_file_name = data_dir + "TNOR3_noise_y.tsv"
    X = pd.read_table(noise_X_file_name)
    y = pd.read_table(noise_y_file_name)

    # split into train/validate/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - percent_train, random_state=seed)
    X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=validate_test_ratio,
                                                              random_state=seed)
    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_validate = np.array(X_validate)
    y_validate = np.array(y_validate)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, X_validate, X_test, y_train, y_validate, y_test


# verified numpy implementation
def mean_residual_noise_np(y_true, y_pred):
    return np.mean(np.sum((y_true == 0) * y_pred, axis=1))


# MSE implementation
def MSE(y, y_est):
    return np.mean(((y - y_est) ** 2))


# counts positions for which residual noise presents
def count_noisy_positions(y, y_est):
    row_noise = np.sum((y == 0) * y_est, axis=1)
    return np.count_nonzero(row_noise)


# function for KerasRegressor that initialize model
def create_model(nodes_number, learning_rate):
    # create model
    model = Sequential()
    model.add(Dense(output_dim=nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(output_dim=4, init='normal', activation='relu'))

    # Compile model
    current_optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='poisson', optimizer=current_optimizer, metrics=['mean_squared_error'])
    return model


# three different scorers for GridSearch
# MSE
def scorer_mse(estimator, X, y):
    y_est = estimator.predict(X)
    return MSE(y, y_est)


# percent of noisy positions
def scorer_percent(estimator, X, y):
    y_est = estimator.predict(X)
    n_faults = count_noisy_positions(y, y_est)
    return 100 % n_faults / float(X.shape[0])


# mean noise level at position
def scorer_mrn(estimator, X, y):
    y_est = estimator.predict(X)
    return mean_residual_noise_np(y, y_est)


# learning rate and nodes number tuning
def grid_search(X, y, seed):
    # define sets of values to try
    lr_set = [0.0001, 0.00025, 0.0005, 0.00075, 0.001]
    nn_set = [50, 150, 250, 350, 500]
    # manual grid search, as sklearn version doesn't allow change outpur_dim parameter
    batch_size = 4096
    nb_epoch = 4
    # initialize result dictionary (3D)
    results = {lr: {nn: {'mse': [], 'percent': [], 'mrn': []} for nn in nn_set} for lr in lr_set}
    # go through the grid
    for learning_rate in lr_set[:2]:
        for nodes_number in nn_set[:2]:
            print 'Now running: learning rate {}, number of nodes {}'.format(learning_rate, nodes_number)
            estimator = KerasRegressor(build_fn=create_model, nodes_number=nodes_number, learning_rate=learning_rate,
                                       nb_epoch=nb_epoch, batch_size=batch_size, verbose=0)
            kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
            # run CV
            # it's impossible to calculate several scores in one run, so we have to do three separate runs
            # MSE
            cv_result = cross_val_score(estimator, X, y, cv=kfold, scoring=scorer_mse)
            results[learning_rate][nodes_number]['mse'] = cv_result
            # percentage of noisy positions
            cv_result = cross_val_score(estimator, X, y, cv=kfold, scoring=scorer_percent)
            results[learning_rate][nodes_number]['percent'] = cv_result
            # mean residual moise
            cv_result = cross_val_score(estimator, X, y, cv=kfold, scoring=scorer_mrn)
            results[learning_rate][nodes_number]['mrn'] = cv_result
    # output results
    output_file_name = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/gridSearch.tsv'
    with open(output_file_name, 'w') as out:
        out.write('learning_rate\tnodes_numbed\tmetric\tvalue\n')
        for lr in results:
            for nn in results[lr]:
                for metric in results[lr][nn]:
                    for value in results[lr][nn][metric]:
                        out.write('{}\t{}\t{}\t{}\n'.format(lr, nn, metric, value))


def main():
    # fix random
    seed = 1214
    np.random.seed(seed)

    # split data into sets
    percent_train = 0.85
    validate_test_ratio = 0.5
    X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(seed, percent_train, validate_test_ratio)

    grid_search(X_train, y_train, seed)

if __name__ == "__main__":
    main()