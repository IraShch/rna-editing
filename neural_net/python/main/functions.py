import keras.backend as K
import pandas as pd
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


# Data preparation

# prepares data for training
def load_training_data(add_noise, data_dir, data_name, scaling_groups_number):
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
        y['A'] /= coverage
        y['C'] /= coverage
        y['G'] /= coverage
        y['T'] /= coverage
    #     y = y.astype(int)

    print y
    return np.array(X), np.array(y)


# prepares data for predictions
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


# model creation and training

# creates model and trains it
def create_model(X_train, y_train, nodes_number, second_layer, batch_size, nb_epoch, opt, scale_in_groups, activation,
                 reg_const, validation_data=None):
    # define model structure
    model = Sequential()
    if scale_in_groups == 2:
        model.add(Dense(nodes_number, input_dim=6, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    else:
        model.add(Dense(nodes_number, input_dim=5, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    if second_layer > 0:
        model.add(Dense(second_layer, init='uniform', activation='tanh', W_regularizer=l2(reg_const)))
    model.add(Dense(4, init='uniform', activation='sigmoid', W_regularizer=l2(reg_const)))
    model.compile(loss='mse', optimizer=opt, metrics=[mean_residual_noise])
    # learn
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0,
                        validation_data=validation_data)
    if validation_data:
        return model, history
    return model


# denoises data
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


# other

# calculates mean noise remained on "zero" positions
# approximation for numpy version
# to be Keras parameter
def mean_residual_noise(y_true, y_pred):
    a = 1
    k = 1000000
    return K.mean(K.sum((a / (k * K.pow(y_true, 2) + 1)) * y_pred, axis=1))


def write_log(activation, batch_size, data_dir, l2_reg_const, log_file, nb_epoch, nodes_number, opt,
              scaling_groups_number, is_test, second_layer_nodes, percent_train=0):
    with open(log_file, 'w') as out:
        out.write('Data directory: {}\n'.format(data_dir))
        if is_test:
            out.write('Percent of data to train model: {}\n'.format(percent_train))
        if second_layer_nodes == 0:
            out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
        else:
            out.write('Number of nodes in the first hidden layer: {}\n'.format(nodes_number))
            out.write('Number of nodes in the second hidden layer: {}\n'.format(second_layer_nodes))
        out.write('Number of epochs: {}\n'.format(nb_epoch))
        out.write('Batch size: {}\n'.format(batch_size))
        out.write('Loss function: MSE\n')
        out.write('Optimizer: {}\n'.format(opt))
        out.write('Scale coverage: {}\n'.format(scaling_groups_number > 0))
        if scaling_groups_number > 0:
            out.write('Split into two groups by coverage: {}\n'.format(scaling_groups_number == 2))
        out.write('Activation function: {}\n'.format(activation))
        out.write('L2 regularisation constant: {}\n'.format(l2_reg_const))