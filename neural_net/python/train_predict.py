import argparse
import os
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


# prepares data for learning
def prepare_training_data(data_dir, data_name):
    # load initial noise set
    noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
    noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X_train = pd.read_table(noise_X_file_name)
    y_train = pd.read_table(noise_y_file_name)

    # all data must be np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


# prepares data for predicting
def prepare_noisy_data(data_dir, data_name, coverage_threshold):
    # load initial noise set
    adar_file_name = '{}{}_adar.tsv'.format(data_dir, data_name)
    apobec_file_name = '{}{}_apobec.tsv'.format(data_dir, data_name)
    snp_file_name = '{}{}_snp.tsv'.format(data_dir, data_name)
    adar = pd.read_table(adar_file_name)
    apobec = pd.read_table(apobec_file_name)
    snp = pd.read_table(snp_file_name)

    # filter by coverage
    adar = adar[adar['coverage'] >= coverage_threshold]
    apobec = apobec[apobec['coverage'] >= coverage_threshold]
    snp = snp[snp['coverage'] >= coverage_threshold]

    adar = adar.loc[:, 'A':'T']
    apobec = apobec.loc[:, 'A':'T']
    snp = snp.loc[:, 'A':'T']

    return np.array(adar), np.array(apobec), np.array(snp)


# creates model and trains it
def create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, output_dir, data_name):
    # define model structure
    model = Sequential()
    model.add(Dense(nodes_number, input_dim=4, init='normal', activation='tanh'))
    model.add(Dense(4, init='normal', activation='relu'))
    model.compile(loss='poisson', optimizer='rmsprop', metrics=['mean_squared_error', mean_residual_noise])
    # learn
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0)
    return model


# denoising predictions
def denoise(model, X, target_name, data_name, output_dir):
    y = model.predict(X)
    output_file_name = output_dir + data_name + '_' + target_name + '_denoised.tsv'
    np.savetxt(output_file_name, y, fmt='%.3f', delimiter='\t', comments='', header='A\tC\tG\tT')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
    parser.add_argument('-o', '--outputDir', help='directory to save the results in', required=True)
    parser.add_argument('-n', '--nodesNumber', help='number of nodes in the hidden layer', default=50)
    parser.add_argument('-b', '--batchSize', help='batch size', default=4096)
    parser.add_argument('-e', '--epochNumber', help='number of epochs', default=400)
    parser.add_argument('-c', '--coverageThreshold', help='min required coverage', default=10)

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

    # prepare directory for the results
    # create directory with all results for this dataset
    out_dir += data_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # create directory with train_test result
    out_dir += '/train_predict_{}nodes_{}epochs'.format(nodes_number, nb_epoch)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    # write logs
    log_file = out_dir + 'log_predict.txt'
    with open(log_file, 'w') as out:
        out.write('Data directory: {}\n'.format(data_dir))
        out.write('Number of nodes in hidden layer: {}\n'.format(nodes_number))
        out.write('Number of epochs: {}\n'.format(nb_epoch))
        out.write('Batch size: {}\n'.format(batch_size))

    # fix random
    seed = 1214
    np.random.seed(seed)

    # split into sets
    X_train, y_train = prepare_training_data(data_dir, data_name)
    # train model
    model = create_model(X_train, y_train, nodes_number, batch_size, nb_epoch, out_dir, data_name)

    # load noisy data
    adar, apobec, snp = prepare_noisy_data(data_dir, data_name, coverage_threshold)
    # denoise data
    denoise(model, adar, 'ADAR', data_name, out_dir)
    denoise(model, apobec, 'APOBEC', data_name, out_dir)
    denoise(model, snp, 'SNP', data_name, out_dir)


if __name__ == "__main__":
    main()
