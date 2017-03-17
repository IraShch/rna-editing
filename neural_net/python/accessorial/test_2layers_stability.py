import argparse
import os
import random
import numpy as np
import pandas as pd
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
def prepare_training_data(data_dir, data_name, include_coverage, train_on_identical, use_fractions,
                          scaling_groups_number, k=1.0):
    # load initial noise set
    if include_coverage:
        noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y_cov.tsv'.format(data_dir, data_name)
    else:
        noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X_train = pd.read_table(noise_X_file_name)
    y_train = pd.read_table(noise_y_file_name)

    # scale coverage
    if include_coverage and scaling_groups_number == 1:
        med_coverage = X_train.median()['coverage']
        X_train['coverage'] /= med_coverage
    if include_coverage and scaling_groups_number == 2:
        X_train['group'] = (X_train['coverage'] > 1000).astype(int)
        penguins = X_train[X_train['group'] == 0].reset_index(drop=True)
        bears = X_train[X_train['group'] == 1].reset_index(drop=True)
        med_coverage = penguins.median()['coverage']
        penguins['coverage'] /= med_coverage
        med_coverage = bears.median()['coverage']
        bears['coverage'] /= med_coverage
        X_train = pd.concat([penguins, bears], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

    if use_fractions:
        coverage = X_train['A'] + X_train['C'] + X_train['G'] + X_train['T']
        X_train['A'] /= coverage
        X_train['C'] /= coverage
        X_train['G'] /= coverage
        X_train['T'] /= coverage

        y_train['A'] = y_train['A'] > 0
        y_train['C'] = y_train['C'] > 0
        y_train['G'] = y_train['G'] > 0
        y_train['T'] = y_train['T'] > 0

        y_train = y_train.astype(int)

    if train_on_identical:
        X_train, y_train = add_identical(X_train, y_train, k)

    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train

# prepares non-standard data for predictions
def prepare_custom_dataset(file_name, coverage_threshold, use_fractions, scale_coverage):
    dataset = pd.read_table(file_name)
    # filter by coverage
    dataset =  dataset[dataset['coverage'] >= coverage_threshold]
    dataset['ini_coverage'] = dataset['coverage']

    if scale_coverage == 1:
        med_coverage = dataset.median()['coverage']
        dataset['coverage'] /= med_coverage
    if scale_coverage == 2:
        dataset['group'] = (dataset['coverage'] > 1000).astype(int)
        penguins = dataset[dataset['group'] == 0].reset_index(drop=True)
        bears = dataset[dataset['group'] == 1].reset_index(drop=True)
        med_coverage = penguins.median()['coverage']
        penguins['coverage'] /= med_coverage
        med_coverage = bears.median()['coverage']
        bears['coverage'] /= med_coverage
        dataset = pd.concat([penguins, bears], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

    if use_fractions:
        dataset['A'] /= dataset['ini_coverage']
        dataset['C'] /= dataset['ini_coverage']
        dataset['G'] /= dataset['ini_coverage']
        dataset['T'] /= dataset['ini_coverage']

    return dataset


# prepares data for predicting
def prepare_noisy_data(data_dir, data_name, coverage_threshold, use_fractions, scale_coverage):
    # load initial noise set
    adar_file_name = '{}{}_adar.tsv'.format(data_dir, data_name)
    apobec_file_name = '{}{}_apobec.tsv'.format(data_dir, data_name)
    snp_file_name = '{}{}_snp.tsv'.format(data_dir, data_name)

    adar = prepare_custom_dataset(adar_file_name, coverage_threshold, use_fractions, scale_coverage)
    apobec = prepare_custom_dataset(apobec_file_name, coverage_threshold, use_fractions, scale_coverage)
    snp = prepare_custom_dataset(snp_file_name, coverage_threshold, use_fractions, scale_coverage)

    return adar, apobec, snp


# creates model and trains it
def create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, include_coverage, loss, opt, scale_in_groups,
                 activation):
    # define model structure
    model = Sequential()
    reg_const = 0.001
    print scale_in_groups
    nodes_number = 40
    if include_coverage and scale_in_groups == 2:
        model.add(Dense(nodes_number, input_dim=6, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    elif include_coverage and scale_in_groups != 2:
        model.add(Dense(nodes_number, input_dim=5, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    else:
        model.add(Dense(nodes_number, input_dim=4, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    nodes_number = 20
    model.add(Dense(nodes_number, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    model.add(Dense(4, init='uniform', activation=activation, W_regularizer=l2(reg_const)))
    model.compile(loss=loss, optimizer=opt, metrics=['mean_squared_error', mean_residual_noise])
    # learn
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    return model


# denoising predictions
def denoise(model, X, target_name, data_name, output_dir, include_coverage, scale_groups):
    old_names = [name for name in X.columns]
    if include_coverage and scale_groups != 2:
        y = model.predict(np.array(X.loc[:, 'A':'coverage']))
    elif scale_groups == 2:
        y = model.predict(np.array(X.loc[:, ['A', 'C', 'G', 'T', 'group', 'coverage']]))
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
    parser.add_argument('-m', '--customFile', help='predict on custom file only', default='')
    parser.add_argument('-j', '--iteration', help='number of run', required=True)
    parser.add_argument('-v', '--includeCoverage', help='include coverage column into X', action='store_true')
    parser.add_argument('-s', '--identicalPositions', help='include identical positions into training set',
                        action='store_true')
    parser.add_argument('-k', '--identicalPercent', help='number of identical positions added to trainig dataset: '
                                                         'k * nrow(X)', default=1)
    parser.add_argument('-l', '--loss', help='specify loss function to use: '
                                             'poisson (default) or mse', default='poisson')
    parser.add_argument('-t', '--optimizer', help='specify optimizer to use: '
                                                  'rmsprop (default) or adam', default='rmsprop')
    parser.add_argument('-f', '--fractions', help='use fractions instead of absolute values', action='store_true')
    parser.add_argument('-a', '--activation', help='output layer activation function: relu (default) or softmax',
                        default='relu')
    parser.add_argument('-g', '--groupsToScale',
                        help='In how many groups split dataset while scaling coverage (0, 1, 2)',
                        default=0)

    args = parser.parse_args()

    data_dir = args.dataDir
    if not data_dir.endswith('/'):
        data_dir += '/'
    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'
    data_name = args.dataName

    use_fractions = args.fractions

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
        if use_fractions:
            raise ValueError('You do not need identical positions while using fractions')
    else:
        train_on_identical = False
        percent_identical = 0

    loss = args.loss
    if loss not in ['poisson', 'mse']:
        raise ValueError('Loss function can be only poisson or mse')
    if use_fractions and loss != 'mse':
        raise ValueError('Loss function can be only mse while using fractions')

    opt = args.optimizer
    if opt not in ['rmsprop', 'adam']:
        raise ValueError('Optimizer can be only rmsprop or adam!')

    activation = args.activation
    if activation not in ['relu', 'softmax']:
        raise ValueError('Activation function can be only relu or softmax!')

    scaling_groups_number = int(args.groupsToScale)
    if scaling_groups_number not in [0, 1, 2]:
        raise ValueError('Number of groups may be only 0, 1 or 2!')

    # prepare directory for the results
    # create directory with all results for this dataset
    out_dir += data_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create directory with train_test result
    additional_string = ''
    if use_fractions:
        additional_string += "_fractions"
    if scaling_groups_number > 0:
        additional_string += '_{}scaling'.format(scaling_groups_number)
    if activation == 'softmax':
        additional_string += '_softmax'
    out_dir += '/train_predict_{}nodes_{}epochs_{}coverage_{}identical_{}loss_{}opt{}'.format(nodes_number, nb_epoch,
                                                                            int(include_coverage),
                                                                            percent_identical,
                                                                            loss,
                                                                            opt,
                                                                            additional_string)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    is_custom = (args.customFile != '')


    # split into sets
    print "Prepare training set"
    X_train, y_train = prepare_training_data(data_dir, data_name, include_coverage, train_on_identical,
                                             use_fractions, scaling_groups_number, percent_identical)
    # train model
    print "Train model"
    model = create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, include_coverage, loss, opt,
                         scaling_groups_number, activation)

    if not is_custom:
        # write logs
        log_file = out_dir + 'log_predict.txt'
        with open(log_file, 'w') as out:
            out.write('Data directory: {}\n'.format(data_dir))
            out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
            out.write('Number of epochs: {}\n'.format(nb_epoch))
            out.write('Batch size: {}\n'.format(batch_size))
            out.write('Loss function: {}\n'.format(loss))
            out.write('Optimizer: {}\n'.format(opt))
            out.write('Input coverage: {}\n'.format(include_coverage))
            out.write('Include identical positions into training set: {}\n'.format(train_on_identical))
            if train_on_identical:
                out.write('Identical rows to be added: {} * nrow(X_train)\n'.format(percent_identical))
            out.write('Use fractions: {}\n'.format(use_fractions))
            out.write('Scale coverage: {}\n'.format(scaling_groups_number > 0))
            if scaling_groups_number > 0:
                out.write('Split into two groups by coverage: {}\n'.format(scaling_groups_number == 2))
            out.write('Activation function: {}\n'.format(activation))

        # load noisy data
        adar, apobec, snp = prepare_noisy_data(data_dir, data_name, coverage_threshold, use_fractions,
                                               scaling_groups_number)
        # denoise data
        denoise(model, adar, 'ADAR', data_name, out_dir, include_coverage, scaling_groups_number)
        denoise(model, apobec, 'APOBEC', data_name, out_dir, include_coverage, scaling_groups_number)
        denoise(model, snp, 'SNP', data_name, out_dir, include_coverage, scaling_groups_number)
    else:
        print "Prepare dataset for predictions"
        dataset = prepare_custom_dataset(args.customFile, coverage_threshold, use_fractions, scaling_groups_number)
        target_name = args.customFile.split('/')[-1].split('.')[0] + str(args.iteration)
        print "Denoise"
        denoise(model, dataset, target_name, data_name, out_dir, include_coverage, scaling_groups_number)


if __name__ == "__main__":
    main()
