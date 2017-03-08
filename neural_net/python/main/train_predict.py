import argparse
import os
import random
import numpy as np
import pandas as pd
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
def prepare_training_data(data_dir, data_name, include_coverage, train_on_identical, k=1.0):
    # load initial noise set
    if include_coverage:
        noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y_cov.tsv'.format(data_dir, data_name)
    else:
        noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X_train = pd.read_table(noise_X_file_name)
    y_train = pd.read_table(noise_y_file_name)

    if train_on_identical:
        X_train, y_train = add_identical(X_train, y_train, k)

    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


# prepares non-standard data for predictiong
def prepare_custom_dataset(file_name, coverage_threshold):
    dataset = pd.read_table(file_name)
    # filter by coverage and return
    return dataset[dataset['coverage'] >= coverage_threshold]


# prepares data for predicting
def prepare_noisy_data(data_dir, data_name, coverage_threshold):
    # load initial noise set
    adar_file_name = '{}{}_adar.tsv'.format(data_dir, data_name)
    apobec_file_name = '{}{}_apobec.tsv'.format(data_dir, data_name)
    snp_file_name = '{}{}_snp.tsv'.format(data_dir, data_name)

    adar = prepare_custom_dataset(adar_file_name, coverage_threshold)
    apobec = prepare_custom_dataset(apobec_file_name, coverage_threshold)
    snp = prepare_custom_dataset(snp_file_name, coverage_threshold)

    return adar, apobec, snp


# creates model and trains it
def create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, include_coverage):
    # define model structure
    model = Sequential()
    if include_coverage:
        model.add(Dense(nodes_number, input_dim=5, init='normal', activation='tanh'))
    else:
        model.add(Dense(nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.compile(loss='poisson', optimizer='rmsprop', metrics=['mean_squared_error', mean_residual_noise])
    # learn
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    return model


# denoising predictions
def denoise(model, X, target_name, data_name, output_dir, include_coverage):
    old_names = [name for name in X.columns]
    if include_coverage:
        y = model.predict(np.array(X.loc[:, 'A':'coverage']))
    else:
        y = model.predict(np.array(X.loc[:, 'A':'T']))
    y = pd.DataFrame(y)
    result_df = pd.concat([X.reset_index(drop=True), y], axis=1, ignore_index=True)
    old_names.extend(['A_pred', 'C_pred', 'G_pred', 'T_pred'])
    result_df.columns = old_names
    output_file_name = output_dir + data_name + '_' + target_name + '_denoised.tsv'
    result_df.to_csv(output_file_name, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
    parser.add_argument('-o', '--outputDir', help='directory to save the results in', required=True)
    parser.add_argument('-n', '--nodesNumber', help='number of nodes in the hidden layer', default=50)
    parser.add_argument('-b', '--batchSize', help='batch size', default=4096)
    parser.add_argument('-e', '--epochNumber', help='number of epochs', default=400)
    parser.add_argument('-c', '--coverageThreshold', help='min required coverage', default=10)
    parser.add_argument('-f', '--customFile', help='predict on custom file only', default='')
    parser.add_argument('-v', '--includeCoverage', help='include coverage column into X', action='store_true')
    parser.add_argument('-s', '--identicalPositions', help='include identical positions into training set',
                        action='store_true')
    parser.add_argument('-k', '--identicalPercent', help='number of identical positions added to trainig dataset: '
                                                         'k * nrow(X)', default=1)

    args = parser.parse_args()

    data_dir = args.dataDir
    if not data_dir.endswith('/'):
        data_dir += '/'
    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'
    data_name = args.dataName

    nodes_number = int(args.nodesNumber)
    batch_size = int(args.batchSize)
    nb_epoch = int(args.epochNumber)
    coverage_threshold = int(args.coverageThreshold)

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


    # prepare directory for the results
    # create directory with all results for this dataset
    out_dir += data_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create directory with train_test result
    out_dir += '/train_predict_{}nodes_{}epochs_{}coverage_{}identical'.format(nodes_number, nb_epoch,
                                                                            int(include_coverage),
                                                                            percent_identical)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    is_custom = (args.customFile != '')

    # fix random
    seed = 1214
    np.random.seed(seed)

    # split into sets
    X_train, y_train = prepare_training_data(data_dir, data_name, include_coverage,
                                             train_on_identical, percent_identical)
    # train model
    model = create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, include_coverage)

    if not is_custom:
        # write logs
        log_file = out_dir + 'log_predict.txt'
        with open(log_file, 'w') as out:
            out.write('Data directory: {}\n'.format(data_dir))
            out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
            out.write('Number of epochs: {}\n'.format(nb_epoch))
            out.write('Batch size: {}\n'.format(batch_size))
            out.write('Input coverage: {}\n'.format(include_coverage))
            out.write('Include identical positions into training set: {}\n'.format(train_on_identical))
            if train_on_identical:
                out.write('Identical rows to be added: {} * nrow(X_train)\n'.format(percent_identical))

        # load noisy data
        adar, apobec, snp = prepare_noisy_data(data_dir, data_name, coverage_threshold)
        # denoise data
        denoise(model, adar, 'ADAR', data_name, out_dir, include_coverage)
        denoise(model, apobec, 'APOBEC', data_name, out_dir, include_coverage)
        denoise(model, snp, 'SNP', data_name, out_dir, include_coverage)
    else:
        dataset = prepare_custom_dataset(args.customFile, coverage_threshold)
        target_name = args.customFile.split('/')[-1].split('.')[0]
        denoise(model, dataset, target_name, data_name, out_dir, include_coverage)




if __name__ == "__main__":
    main()
