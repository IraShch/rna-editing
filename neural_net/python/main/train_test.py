import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from functions import *


# prepares data for learning
def split_data(data_dir, data_name, percent_train, seed, scaling_groups_number, add_noise):
    X, y = load_training_data(add_noise, data_dir, data_name, scaling_groups_number)

    # split into train/validate/test
    if percent_train == 1:
        return X, X, y, y

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


def test(activation, add_noise, batch_size, data_dir, data_name, l2_reg_const, nb_epoch, nodes_number, opt, out_dir,
         percent_train, plots_dir, scaling_groups_number, seed, second_layer):
    # split data into sets
    X_train, X_test, y_train, y_test = split_data(data_dir, data_name, percent_train, seed, scaling_groups_number,
                                                  add_noise)
    # run
    model, history = create_model(X_train, y_train, nodes_number, second_layer, batch_size, nb_epoch, opt, scaling_groups_number,
                                  activation, l2_reg_const, validation_data=(X_test, y_test))
    show_learning_history(history, data_name, plots_dir)
    test_model(model, X_test, y_test, data_name, out_dir)


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
    parser.add_argument('-w', '--secondLayer', help='how many nodes in the second hiden layer', default=0)

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

    l2_reg_const = float(args.lTwoConstant)
    add_noise = float(args.addNoise)
    second_layer_nodes = int(args.secondLayer)

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
    if second_layer_nodes > 0:
        additional_string += '_second{}'.format(second_layer_nodes)
    out_dir += '/train_test_{}nodes_{}epochs_{}percent{}'.format(nodes_number, nb_epoch, percent_train,
                                                                 additional_string)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'
    # create directory for history plots
    plots_dir = out_dir + '/history_plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plots_dir += '/'

    # write logs
    log_file = out_dir + 'log_test.txt'
    write_log(activation, batch_size, data_dir, l2_reg_const, log_file, nb_epoch, nodes_number, opt,
              scaling_groups_number, True, second_layer_nodes, percent_train)

    # fix random
    seed = 1214
    np.random.seed(seed)

    test(activation, add_noise, batch_size, data_dir, data_name, l2_reg_const, nb_epoch, nodes_number, opt, out_dir,
         percent_train, plots_dir, scaling_groups_number, seed, second_layer_nodes)




if __name__ == "__main__":
    main()
