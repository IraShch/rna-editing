import argparse
import os
from functions import *


def predict(activation, add_noise, args, batch_size, coverage_threshold, data_dir, data_name, is_custom, l2_reg_const,
            nb_epoch, nodes_number, opt, out_dir, scaling_groups_number, second_layer_nodes):
    # split into sets
    X_train, y_train = load_training_data(add_noise, data_dir, data_name, scaling_groups_number)
    # train model
    model = create_model(X_train, y_train, nodes_number, second_layer_nodes, batch_size, nb_epoch, opt,
                         scaling_groups_number, activation, l2_reg_const)
    if not is_custom:
        # write logs
        log_file = out_dir + 'log_predict.txt'
        write_log(activation, batch_size, data_dir, l2_reg_const, log_file, nb_epoch, nodes_number, opt,
                  scaling_groups_number, False, second_layer_nodes)

        # load noisy data
        adar_file_name = '{}{}_adar.tsv'.format(data_dir, data_name)
        apobec_file_name = '{}{}_apobec.tsv'.format(data_dir, data_name)
        snp_file_name = '{}{}_snp.tsv'.format(data_dir, data_name)

        adar = prepare_custom_dataset(adar_file_name, coverage_threshold, scaling_groups_number)
        apobec = prepare_custom_dataset(apobec_file_name, coverage_threshold, scaling_groups_number)
        snp = prepare_custom_dataset(snp_file_name, coverage_threshold, scaling_groups_number)

        # denoise data
        denoise(model, adar, 'ADAR', data_name, out_dir, scaling_groups_number)
        denoise(model, apobec, 'APOBEC', data_name, out_dir, scaling_groups_number)
        denoise(model, snp, 'SNP', data_name, out_dir, scaling_groups_number)
    else:
        dataset = prepare_custom_dataset(args.customFile, coverage_threshold, scaling_groups_number)
        target_name = args.customFile.split('/')[-1].split('.')[0]
        denoise(model, dataset, target_name, data_name, out_dir, scaling_groups_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
    parser.add_argument('-o', '--outputDir', help='directory to save the results in', required=True)
    parser.add_argument('-n', '--nodesNumber', help='number of nodes in the hidden layer', default=50)
    parser.add_argument('-b', '--batchSize', help='batch size', default=4096)
    parser.add_argument('-e', '--epochNumber', help='number of epochs', default=400)
    parser.add_argument('-c', '--coverageThreshold', help='min required coverage', default=10)
    parser.add_argument('-a', '--customFile', help='predict on custom file only', default='')
    parser.add_argument('-t', '--optimizer', help='specify optimizer to use: '
                                                 'rmsprop (default) or adam', default='rmsprop')
    parser.add_argument('-a', '--activation', help='output layer activation function: relu (default) or softmax',
                        default='relu')
    parser.add_argument('-g', '--groupsToScale',
                        help='In how many groups split dataset while scaling coverage (0, 1, 2)',
                        default=0)
    parser.add_argument('-s', '--addNoise', help='add some noise into Y training', action='store_true')
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
    out_dir += '/train_predict_{}nodes_{}epochs{}'.format(nodes_number, nb_epoch, additional_string)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir += '/'

    is_custom = (args.customFile != '')

    # fix random
    seed = 1214
    np.random.seed(seed)

    predict(activation, add_noise, args, batch_size, coverage_threshold, data_dir, data_name, is_custom, l2_reg_const,
            nb_epoch, nodes_number, opt, out_dir, scaling_groups_number, second_layer_nodes)


if __name__ == "__main__":
    main()
