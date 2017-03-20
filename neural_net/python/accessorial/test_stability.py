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


# prepares data for learning
def prepare_training_data(data_dir, data_name, scaling_groups_number, add_noise):
    # load initial noise set
    noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
    noise_y_file_name = '{}{}_noise_y_cov.tsv'.format(data_dir, data_name)
    X_train = pd.read_table(noise_X_file_name)
    y_train = pd.read_table(noise_y_file_name)

    # scale coverage
    if scaling_groups_number == 1:
        med_coverage = X_train.median()['coverage']
        X_train['coverage'] /= med_coverage
    if scaling_groups_number == 2:
        X_train['group'] = (X_train['coverage'] > 1000).astype(int)
        penguins = X_train[X_train['group'] == 0].reset_index(drop=True)
        bears = X_train[X_train['group'] == 1].reset_index(drop=True)
        med_coverage = penguins.median()['coverage']
        penguins['coverage'] /= med_coverage
        med_coverage = bears.median()['coverage']
        bears['coverage'] /= med_coverage
        X_train = pd.concat([penguins, bears], axis=0, ignore_index=True).sample(frac=1).reset_index(drop=True)

    coverage = X_train['A'] + X_train['C'] + X_train['G'] + X_train['T']
    X_train['A'] /= coverage
    X_train['C'] /= coverage
    X_train['G'] /= coverage
    X_train['T'] /= coverage

    if add_noise > 0:
        main_part = random.uniform(add_noise, 1)
        additional = (1 - main_part) / float(3)
        y_train['A'] = (y_train['A'] > 0) * (main_part - additional) + additional
        y_train['C'] = (y_train['C'] > 0) * (main_part - additional) + additional
        y_train['G'] = (y_train['G'] > 0) * (main_part - additional) + additional
        y_train['T'] = (y_train['T'] > 0) * (main_part - additional) + additional
    else:
        y_train['A'] = y_train['A'] > 0
        y_train['C'] = y_train['C'] > 0
        y_train['G'] = y_train['G'] > 0
        y_train['T'] = y_train['T'] > 0
        y_train = y_train.astype(int)

    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


# prepares non-standard data for predictions
def prepare_custom_dataset(file_name, coverage_threshold, scale_coverage):
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

    dataset['A'] /= dataset['ini_coverage']
    dataset['C'] /= dataset['ini_coverage']
    dataset['G'] /= dataset['ini_coverage']
    dataset['T'] /= dataset['ini_coverage']

    return dataset


# creates model and trains it
def create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, opt, scale_in_groups, activation, reg_const):
    # define model structure
    model = Sequential()
    print scale_in_groups
    if scale_in_groups == 2:
        model.add(Dense(nodes_number, input_dim=6, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    else:
        model.add(Dense(nodes_number, input_dim=5, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    model.add(Dense(4, init='uniform', activation=activation, W_regularizer=l2(reg_const)))
    model.compile(loss='mse', optimizer=opt, metrics=[mean_residual_noise])
    # learn
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    return model


# denoising predictions
def denoise(model, X, target_name, data_name, output_dir, scale_groups):
    old_names = [name for name in X.columns]
    if scale_groups != 2:
        y = model.predict(np.array(X.loc[:, 'A':'coverage']))
    else:
        y = model.predict(np.array(X.loc[:, ['A', 'C', 'G', 'T', 'group', 'coverage']]))
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
    parser.add_argument('-m', '--customFile', help='predict on custom file only', required=True)
    parser.add_argument('-j', '--iteration', help='number of run', required=True)
    parser.add_argument('-t', '--optimizer', help='specify optimizer to use: '
                                                  'rmsprop (default) or adam', default='rmsprop')
    parser.add_argument('-a', '--activation', help='output layer activation function: relu (default) or softmax',
                        default='relu')
    parser.add_argument('-g', '--groupsToScale',
                        help='In how many groups split dataset while scaling coverage (0, 1, 2)',
                        default=0)
    parser.add_argument('-s', '--addNoise', help='add some noise into Y training', action='store_true')
    parser.add_argument('-l', '--lTwoConstant', help='l2 regularisation constant', default=0)

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

    opt = args.optimizer
    if opt not in ['rmsprop', 'adam']:
        raise ValueError('Optimizer can be only rmsprop or adam!')

    activation = args.activation
    if activation not in ['relu', 'softmax']:
        raise ValueError('Activation function can be only relu or softmax!')

    scaling_groups_number = int(args.groupsToScale)
    if scaling_groups_number not in [0, 1, 2]:
        raise ValueError('Number of groups may be only 0, 1 or 2!')

    l2_reg_const = float(args.lTwoConstant)
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
    out_dir += '/train_predict_{}nodes_{}epochs{}'.format(nodes_number, nb_epoch, additional_string)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    # split into sets
    X_train, y_train = prepare_training_data(data_dir, data_name, scaling_groups_number, add_noise)
    # train model
    model = create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, opt, scaling_groups_number, activation,
                         l2_reg_const)
    dataset = prepare_custom_dataset(args.customFile, coverage_threshold, scaling_groups_number)
    target_name = args.customFile.split('/')[-1].split('.')[0] + str(args.iteration)
    denoise(model, dataset, target_name, data_name, out_dir, scaling_groups_number)



if __name__ == "__main__":
    main()
