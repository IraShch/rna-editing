import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
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
    noise_X_file_name = data_dir + "TNOR3/TNOR3_noise_X.tsv"
    noise_y_file_name = data_dir + "TNOR3/TNOR3_noise_y.tsv"
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


# calculates mean noise remained on "zero" positions
# approximation for numpy version
# to be Keras parameter
def mean_residual_noise(y_true, y_pred):
    a = 1
    k = 1000000
    return K.mean(K.sum((a / (k * K.pow(y_true, 2) + 1)) * y_pred, axis=1))


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


# counts positions for which max value
# corresponds to non-reference nucleotide
def count_fault(y, y_est):
    a = np.sum(y, axis=1)
    a = np.reshape(a, (a.shape[0], 1))
    b = (y == 0) * y_est
    return np.count_nonzero(np.sum(b > a, axis=1))


# visualize learning history
def show_learning_history(history, save=False, fname=None):
    # loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    if save:
        plt.savefig(fname + '_loss.png')
        plt.clf()
    else:
        plt.show()
    # MSE
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    if save:
        plt.savefig(fname + '_mse.png')
        plt.clf()
    else:
        plt.show()
    # residual noise for zero positions
    plt.plot(history.history['mean_residual_noise'])
    plt.plot(history.history['val_mean_residual_noise'])
    plt.title('Residual noise')
    plt.ylabel('noise')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    if save:
        plt.savefig(fname + '_mrn.png')
        plt.clf()
    else:
        plt.show()


# evaluate model
def diagnose_model(model, X, y):
    y_est = model.predict(X)
    np.set_printoptions(precision=1, formatter={'float_kind': lambda x: "%.1f" % x})

    # basic stats
    total_positions = X.shape[0]
    print "\nModel diagnostic:"
    print("MSE: {}".format(MSE(y, y_est)))
    print "Mean residual noise: {}".format(mean_residual_noise_np(y, y_est))
    n_noisy = count_noisy_positions(y, y_est)
    print "Positions with residual noise: {} out of {} ({}%)".format(n_noisy, total_positions,
                                                                     100 * n_noisy / float(total_positions))
    n_faults = count_fault(y, y_est)
    print "Rows for which incorrect nucleotide has maximum value: " \
          "{} out of {} ({}%)".format(n_faults, total_positions, 100 % n_faults / float(total_positions))
    print ""

    # show data
    print "Estimated values:\n", y_est
    print "True values:\n", y
    print "Input:\n", X


# plot histograms of different metrics:
# y_est / (y + 1) for reference nucleotides
# amount of residual noise in positions (row sum)
# amount of residual noise per nucleotide (without sum)
# for positions with noise: value of noise / (real value + 1)
def mistakes_hists(model, X, y):
    y_est = model.predict(X)

    # reference nucleotide
    plt.hist(np.sum((y != 0) * (y_est / (y + 1)), axis=1), color='#009688', bins=300)
    plt.title('Reference positions error: n_est / (n + 1)')
    plt.show()

    # TODO: fix scale
    # zero nucleotides
    residual_noise = (y == 0) * y_est
    residual_noise_sum = np.sum(residual_noise, axis=1)
    plt.hist(residual_noise_sum, color='#009688', bins=300)
    plt.title('Residual noise per position')
    plt.show()
    plt.hist(np.ravel(residual_noise), color='#009688', bins=300)
    plt.title('All residual noise')
    plt.show()

    # ~signal-to-noise ratio
    noisy_rows_i = np.where(residual_noise_sum > 0)
    y_noisy = np.sum(y[noisy_rows_i], axis=1)
    noise_noisy = np.sum(residual_noise[noisy_rows_i], axis=1)
    plt.hist(noise_noisy / (y_noisy + 1), color='#009688', bins=300)
    plt.title('true nucleotide count / noise')
    plt.show()


# function for simple tuning of #nodes
# for a fixed range of values trains a model, evaluates it on validation set,
# reports statistics
def node_number_tune(X_train, y_train, X_validate, y_validate):
    output_dir = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/plots/nodes_tune/'
    n_range = [10, 50, 100, 500, 1000, 1500]
    batch_size = 4096
    nb_epoch = 400
    mse = []
    mrn = []
    rnp = []
    for node_number in n_range:
        model = Sequential()
        model.add(Dense(node_number, input_dim=4, init='normal', activation='tanh'))
        model.add(Dense(4, init='normal', activation='relu'))
        model.summary()
        model.compile(loss='poisson', optimizer='rmsprop', metrics=['mean_squared_error', mean_residual_noise])
        history = model.fit(X_train, y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_validate, y_validate))
        show_learning_history(history, True, output_dir + str(node_number))

        y_est = model.predict(X_validate)
        # basic stats
        total_positions = X_validate.shape[0]
        print "\n{} nodes\nModel diagnostic:".format(node_number)
        current_mse = MSE(y_validate, y_est)
        mse.append(current_mse)
        print("MSE: {}".format(current_mse))
        current_mrn = mean_residual_noise_np(y_validate, y_est)
        mrn.append(current_mrn)
        print "Mean residual noise: {}".format(current_mrn)
        n_noisy = count_noisy_positions(y_validate, y_est)
        rnp.append(100 * n_noisy / float(total_positions))
        print "Positions with residual noise: {} out of {} ({}%)".format(n_noisy, total_positions,
                                                                         100 * n_noisy / float(total_positions))
    # summary plots
    plt.plot(mse)
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('# nodes')
    plt.savefig(output_dir + 'mse.png')
    plt.clf()
    plt.plot(mrn)
    plt.title('Mean residual noise')
    plt.ylabel('MRN')
    plt.xlabel('# nodes')
    plt.savefig(output_dir + 'mrn.png')
    plt.clf()
    plt.plot(rnp)
    plt.title('% of positions with RN')
    plt.ylabel('%RNP')
    plt.xlabel('# nodes')
    plt.savefig(output_dir + 'rnp.png')
    plt.clf()


# function for simple tuning of learning rate
# for a fixed range of values trains a model, evaluates it on validation set,
# reports statistics
def learning_rate_tune(X_train, y_train, X_validate, y_validate):
    output_dir = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/plots/lr_tune/'
    node_number = 150
    batch_size = 4096
    nb_epoch = 400
    rate_range = [0.0001, 0.001, 0.01, 0.1, 1]
    mse = []
    mrn = []
    rnp = []
    for learning_rate in rate_range:
        model = Sequential()
        model.add(Dense(node_number, input_dim=4, init='normal', activation='tanh'))
        model.add(Dense(4, init='normal', activation='relu'))
        model.summary()
        current_optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(loss='poisson', optimizer=current_optimizer, metrics=['mean_squared_error', mean_residual_noise])
        history = model.fit(X_train, y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_validate, y_validate))
        show_learning_history(history, True, output_dir + str(learning_rate))

        y_est = model.predict(X_validate)
        # basic stats
        total_positions = X_validate.shape[0]
        print "\n{} nodes\nModel diagnostic:".format(node_number)
        current_mse = MSE(y_validate, y_est)
        mse.append(current_mse)
        print("MSE: {}".format(current_mse))
        current_mrn = mean_residual_noise_np(y_validate, y_est)
        mrn.append(current_mrn)
        print "Mean residual noise: {}".format(current_mrn)
        n_noisy = count_noisy_positions(y_validate, y_est)
        rnp.append(100 * n_noisy / float(total_positions))
        print "Positions with residual noise: {} out of {} ({}%)".format(n_noisy, total_positions,
                                                                         100 * n_noisy / float(total_positions))
    # summary plots
    plt.plot(mse)
    plt.title('MSE')
    plt.ylabel('MSE')
    plt.xlabel('learning rate')
    plt.savefig(output_dir + 'mse.png')
    plt.clf()
    plt.plot(mrn)
    plt.title('Mean residual noise')
    plt.ylabel('MRN')
    plt.xlabel('learning rate')
    plt.savefig(output_dir + 'mrn.png')
    plt.clf()
    plt.plot(rnp)
    plt.title('% of positions with RN')
    plt.ylabel('%RNP')
    plt.xlabel('learning rate')
    plt.savefig(output_dir + 'rnp.png')
    plt.clf()


# function for KerasRegressor that initialize model
def create_model(nodes_number, learning_rate):
    # create model
    model = Sequential()
    model.add(Dense(output_dim=nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(output_dim=4, init='normal', activation='relu'))

    # Compile model
    current_optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='poisson', optimizer=current_optimizer, metrics=['mean_squared_error', mean_residual_noise])
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


# evaluate model
def diagnose_model(model, X, y):
    y_est = model.predict(X)
    np.set_printoptions(precision=1, formatter={'float_kind': lambda x: "%.1f" % x})

    # basic stats
    total_positions = X.shape[0]
    print "\nModel diagnostic:"
    print("MSE: {}".format(MSE(y, y_est)))
    print "Mean residual noise: {}".format(mean_residual_noise_np(y, y_est))
    n_noisy = count_noisy_positions(y, y_est)
    print "Positions with residual noise: {} out of {} ({}%)".format(n_noisy, total_positions,
                                                                     100 * n_noisy / float(total_positions))
    n_faults = count_fault(y, y_est)
    print "Rows for which incorrect nucleotide has maximum value: " \
          "{} out of {} ({}%)".format(n_faults, total_positions, 100 % n_faults / float(total_positions))
    print ""

    # show data
    print "Estimated values:\n", y_est
    print "True values:\n", y
    print "Input:\n", X


# final model evaluation
def test_model(model, X, y, model_name):
    y_est = model.predict(X)
    squared_error = (y - y_est) ** 2
    residual_noise_sum = np.sum((y == 0) * y_est, axis=1)
    reference_position_noise = np.sum((y != 0) * (y - y_est), axis=1)
    result_table = np.concatenate((y, y_est, squared_error,
                                   np.column_stack((residual_noise_sum, reference_position_noise))), axis=1)
    output_file_name = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/' + model_name + "_test.tsv"
    np.savetxt(output_file_name, result_table, fmt='%.3f', delimiter='\t', comments='',
               header='A\tC\tG\tT\tA_est\tC_est\tG_est\tT_est\tA_se\tC_se\tG_se\tT_se\tresidual_noise\treference_error')


def train_test(X_train, X_validate, X_test, y_train, y_validate, y_test):
    # don't need validation set: train on it
    X_train = np.concatenate((X_train, X_validate), axis=0)
    y_train = np.concatenate((y_train, y_validate), axis=0)

    # define model structure
    model = Sequential()
    model.add(Dense(50, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.summary()
    model.compile(loss='poisson', optimizer='rmsprop', metrics=['mean_squared_error', mean_residual_noise])

    batch_size = 4096
    nb_epoch = 500
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=2, validation_data=(X_validate, y_validate))

    # test
    output_dir = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/plots/test_clean_50/'
    show_learning_history(history, save=True, fname=output_dir + '50nodes_learning_clean')
    test_model(model, X_test, y_test, "50nodes_clean")


# initial code for model testing
def simple_model_test(X_train, y_train, X_validate, y_validate):
    # define model structure
    model = Sequential()
    model.add(Dense(50, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.summary()
    model.compile(loss='poisson', optimizer='adam', metrics=['mean_squared_error', mean_residual_noise])

    batch_size = 4096
    nb_epoch = 400
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=2, validation_data=(X_validate, y_validate))

    for layer in model.layers:
        weights = layer.get_weights()
        print weights[0]
        print weights[1]

    # show_learning_history(history)
    # diagnose_model(model, X_validate, y_validate)



def main():
    # fix random
    seed = 1214
    np.random.seed(seed)

    ini_file_name = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/BT20/BT20_N1.tsv'
    data = pd.read_table(ini_file_name)

    all_adar = data[(data['can_be_ADAR_editing'] == True)]
    pure_adar = data[(data['can_be_ADAR_editing'] == True) & (data['in_dbsnp'] == False)]

    all_adar.to_csv('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/BT20/BT20_N1_all_adar.tsv',
                    sep='\t', index=False)
    pure_adar.to_csv('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/BT20/BT20_N1_pure_adar.tsv',
                    sep='\t', index=False)



    # split into sets
    # percent_train = 0.85
    # validate_test_ratio = 0.5
    # X_train, X_validate, X_test, y_train, y_validate, y_test = split_data(seed, percent_train, validate_test_ratio)

    # simple_model_test(X_train, y_train, X_validate, y_validate)


if __name__ == "__main__":
    main()
