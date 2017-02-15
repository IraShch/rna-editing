import pandas as pd
import argparse


def adar_fractions(path, data_name):
    # read in data
    file_name_results = path + data_name + '_ADAR_denoised.tsv'
    data = pd.read_table(file_name_results)
    data_pos = data[data['strand'] == '+'].reset_index(drop=True)
    data_neg = data[data['strand'] == '-'].reset_index(drop=True)

    # calculate fractions
    old_fraction_positive = data_pos['G'] / (data_pos['G'] + data_pos['A'])
    old_fraction_negative = data_neg['C'] / (data_neg['C'] + data_neg['T'])
    clean_fraction_positive = data_pos['G_pred'] / (data_pos['G_pred'] + data_pos['A_pred'])
    clean_fraction_negative = data_neg['C_pred'] / (data_neg['C_pred'] + data_neg['T_pred'])

    # add columns, concatenate two datasets
    data_pos = pd.concat([data_pos, old_fraction_positive, clean_fraction_positive], axis=1, ignore_index=True)
    data_neg = pd.concat([data_neg, old_fraction_negative, clean_fraction_negative], axis=1, ignore_index=True)
    data = pd.concat([data_pos, data_neg], axis=0, ignore_index=True)
    data.columns = ['seqnames', 'pos', 'strand', 'reference', 'A', 'C', 'G', 'T', 'coverage',
                    'A_pred', 'C_pred', 'G_pred', 'T_pred', 'fraction_ini', 'fraction_clean']

    #  save data
    output_file_name = path + 'ADAR_fractions.tsv'
    data.to_csv(output_file_name, sep='\t', index=False)

    # analysis
    n_before = sum(old_fraction_positive > 0) + sum(old_fraction_negative > 0)
    n_after = data[data['fraction_clean'] > 0].shape[0]
    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# ADAR sites\n')
        out.write('Number of potential sites: {}\n'.format(n_before))
        out.write('Number of potential sites after denoising: {} ({}% left)\n'.format(n_after,
                                                                                      100 * n_after / float(n_before)))

def apobec_fractions(path, data_name):
    # read in data
    file_name_results = path + data_name + '_APOBEC_denoised.tsv'
    data = pd.read_table(file_name_results)
    data_pos = data[data['strand'] == '+'].reset_index(drop=True)
    data_neg = data[data['strand'] == '-'].reset_index(drop=True)

    # calculate fractions
    old_fraction_positive = data_pos['T'] / (data_pos['T'] + data_pos['C'])
    old_fraction_negative = data_neg['A'] / (data_neg['A'] + data_neg['G'])
    clean_fraction_positive = data_pos['T_pred'] / (data_pos['T_pred'] + data_pos['C_pred'])
    clean_fraction_negative = data_neg['A_pred'] / (data_neg['A_pred'] + data_neg['G_pred'])

    # add columns, concatenate two datasets
    data_pos = pd.concat([data_pos, old_fraction_positive, clean_fraction_positive], axis=1, ignore_index=True)
    data_neg = pd.concat([data_neg, old_fraction_negative, clean_fraction_negative], axis=1, ignore_index=True)
    data = pd.concat([data_pos, data_neg], axis=0, ignore_index=True)
    data.columns = ['seqnames', 'pos', 'strand', 'reference', 'A', 'C', 'G', 'T', 'coverage',
                    'A_pred', 'C_pred', 'G_pred', 'T_pred', 'fraction_ini', 'fraction_clean']

    #  save data
    output_file_name = path + 'APOBEC_fractions.tsv'
    data.to_csv(output_file_name, sep='\t', index=False)

    # analysis
    n_before = sum(old_fraction_positive > 0) + sum(old_fraction_negative > 0)
    n_after = data[data['fraction_clean'] > 0].shape[0]
    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# APOBEC sites\n')
        out.write('Number of potential sites: {}\n'.format(n_before))
        out.write('Number of potential sites after denoising: {} ({}% left)\n'.format(n_after,
                                                                                      100 * n_after / float(n_before)))


# replace nucleotides with complimentary
def invert(x, type_i, strand_i):
    strand = x[strand_i]
    type = x[type_i]
    pairs = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    if strand == '-':
        return pairs[type[0]] + pairs[type[1]]
    return type


# defines type of mismatch (max non-reference nucleotide in the initial dataset)
def choose_type(x, reference_i, A_i, C_i, G_i, T_i):
    reference = x[reference_i]
    A = x[A_i]
    C = x[C_i]
    G = x[G_i]
    T = x[T_i]
    tmp_dict = {'A': A, 'C': C, 'G': G, 'T': T}
    tmp_dict.pop(reference, None)
    second = max(tmp_dict, key=tmp_dict.get)
    return reference + second


def old_fraction(x, type_i, A_i, C_i, G_i, T_i):
    type = x[type_i]
    A = x[A_i]
    C = x[C_i]
    G = x[G_i]
    T = x[T_i]
    tmp_dict = {'A': A, 'C': C, 'G': G, 'T': T}
    return tmp_dict[type[1]] / float(tmp_dict[type[1]] + tmp_dict[type[0]])


def clean_fraction(x, type_i, Apred_i, Cpred_i, Gpred_i, Tpred_i):
    type = x[type_i]
    A = x[Apred_i]
    C = x[Cpred_i]
    G = x[Gpred_i]
    T = x[Tpred_i]
    tmp_dict = {'A': A, 'C': C, 'G': G, 'T': T}
    coverage = float(tmp_dict[type[1]] + tmp_dict[type[0]])
    if coverage == 0:
        return None
    return tmp_dict[type[1]] / coverage


def snp_fractions(path, data_name):
    # read in data
    file_name_results = path + data_name + '_SNP_denoised.tsv'
    data = pd.read_table(file_name_results)

    # choose type and calculate fractions
    data = pd.concat([data, data.apply(choose_type, axis=1)], axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(old_fraction, axis=1)], axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(clean_fraction, axis=1)], axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(invert, axis=1)], axis=1, ignore_index=True)
    data = data.drop(13, axis=1)
    data.columns = ['seqnames', 'pos', 'strand', 'reference', 'A', 'C', 'G', 'T', 'coverage',
                    'A_pred', 'C_pred', 'G_pred', 'T_pred', 'fraction_ini', 'fraction_clean', 'type']

    #  save data
    output_file_name = path + 'SNP_fractions.tsv'
    data.to_csv(output_file_name, sep='\t', index=False)

    # analysis
    n_before = sum(data['fraction_ini'] > 0)
    n_after = sum(data['fraction_clean'] > 0)
    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# SNP sites\n')
        out.write('Number of potential sites: {}\n'.format(n_before))
        out.write('Number of potential sites after denoising: {} ({}% left)\n'.format(n_after,
                                                                                      100 * n_after / float(n_before)))


def custom_fractions(path, tag, file_name):
    data = pd.read_table(file_name)
    old_names = [name for name in data.columns]

    # choose type and calculate fractions
    ref_i = data.columns.get_loc('reference')
    strand_i = data.columns.get_loc('strand')
    A_i = data.columns.get_loc('A')
    C_i = data.columns.get_loc('C')
    G_i = data.columns.get_loc('G')
    T_i = data.columns.get_loc('T')
    Apred_i = data.columns.get_loc('A_pred')
    Cpred_i = data.columns.get_loc('C_pred')
    Gpred_i = data.columns.get_loc('G_pred')
    Tpred_i = data.columns.get_loc('T_pred')
    data = pd.concat([data, data.apply(lambda x: choose_type(x, ref_i, A_i, C_i, G_i, T_i), axis=1)],
                     axis=1, ignore_index=True)
    type_i = data.shape[1] - 1
    data = pd.concat([data, data.apply(lambda x: old_fraction(x, type_i, A_i, C_i, G_i, T_i), axis=1)],
                     axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(lambda x: clean_fraction(x, type_i, Apred_i, Cpred_i, Gpred_i, Tpred_i), axis=1)],
                     axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(lambda x: invert(x, type_i, strand_i), axis=1)], axis=1, ignore_index=True)
    data = data.drop(type_i, axis=1)
    old_names.extend(['fraction_ini', 'fraction_clean', 'type'])
    data.columns = old_names

    #  save data
    output_file_name = path + tag + '_fractions.tsv'
    data.to_csv(output_file_name, sep='\t', index=False)

    # analysis
    n_before = sum(data['fraction_ini'] > 0)
    n_after = sum(data['fraction_clean'] > 0)
    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# {} sites\n'.format(tag))
        out.write('Number of potential sites: {}\n'.format(n_before))
        out.write('Number of potential sites after denoising: {} ({}% left)\n'.format(n_after,
                                                                                      100 * n_after / float(n_before)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
    parser.add_argument('-f', '--customFile', help='predict on custom file only', default='')

    args = parser.parse_args()

    path = args.dataDir
    if not path.endswith('/'):
        path += '/'
    tag = args.dataName
    is_custom = (args.customFile != '')


    if not is_custom:
        adar_fractions(path, tag)
        apobec_fractions(path, tag)
        snp_fractions(path, tag)
    else:
        tag = args.customFile.split('/')[-1].split('.')[0]
        custom_fractions(path, tag, args.customFile)


if __name__ == "__main__":
    main()
