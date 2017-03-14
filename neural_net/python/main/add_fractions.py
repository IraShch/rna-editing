import pandas as pd
import argparse


def calculate_adar_fractions(data_neg, data_pos, old_names, is_fractions):
    if is_fractions:
        old_fraction_positive = data_pos['G']
        old_fraction_negative = data_neg['C']
        clean_fraction_positive = data_pos['G_pred']
        clean_fraction_negative = data_neg['C_pred']
    else:
        # calculate fractions
        old_fraction_positive = data_pos['G'] / (data_pos['G'] + data_pos['A'])
        old_fraction_negative = data_neg['C'] / (data_neg['C'] + data_neg['T'])
        clean_fraction_positive = data_pos['G_pred'] / (data_pos['G_pred'] + data_pos['A_pred'])
        clean_fraction_negative = data_neg['C_pred'] / (data_neg['C_pred'] + data_neg['T_pred'])

    # add columns, concatenate two datasets
    data_pos = pd.concat([data_pos, old_fraction_positive, clean_fraction_positive], axis=1, ignore_index=True)
    data_neg = pd.concat([data_neg, old_fraction_negative, clean_fraction_negative], axis=1, ignore_index=True)
    data = pd.concat([data_pos, data_neg], axis=0, ignore_index=True)
    old_names.extend(['fraction_ini', 'fraction_clean'])
    data.columns = old_names

    return data


def calculate_apobec_fractions(data_neg, data_pos, old_names, is_fractions):
    if is_fractions:
        old_fraction_positive = data_pos['T']
        old_fraction_negative = data_neg['A']
        clean_fraction_positive = data_pos['T_pred']
        clean_fraction_negative = data_neg['A_pred']
    else:
        # calculate fractions
        old_fraction_positive = data_pos['T'] / (data_pos['T'] + data_pos['C'])
        old_fraction_negative = data_neg['A'] / (data_neg['A'] + data_neg['G'])
        clean_fraction_positive = data_pos['T_pred'] / (data_pos['T_pred'] + data_pos['C_pred'])
        clean_fraction_negative = data_neg['A_pred'] / (data_neg['A_pred'] + data_neg['G_pred'])

    # add columns, concatenate two datasets
    data_pos = pd.concat([data_pos, old_fraction_positive, clean_fraction_positive], axis=1, ignore_index=True)
    data_neg = pd.concat([data_neg, old_fraction_negative, clean_fraction_negative], axis=1, ignore_index=True)
    data = pd.concat([data_pos, data_neg], axis=0, ignore_index=True)
    old_names.extend(['fraction_ini', 'fraction_clean'])
    data.columns = old_names

    return data


def save_and_log(data, path, set_name):
    #  save data
    output_file_name = path + '{}_fractions.tsv'.format(set_name)
    data.to_csv(output_file_name, sep='\t', index=False)

    # analysis
    n_before = data.shape[0]
    n_after = data[data['fraction_clean'] > 0].shape[0]
    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# {} sites\n'.format(set_name))
        out.write('Initial number of sites: {}\n'.format(n_before))
        out.write('After denoising: {} ({}% left)\n'.format(n_after, 100 * n_after / float(n_before)))


def add_adar_fractions(path, data_name, is_strand_specific, is_fractions, file_name_results=None):
    # read in data
    if not file_name_results:
        file_name_results = path + data_name + '_ADAR_denoised.tsv'
        tag = "ADAR"
    else:
        tag = file_name_results.split('/')[-1].split('.')[0] + '_ADAR'
    data = pd.read_table(file_name_results)
    old_names = [name for name in data.columns]

    # add fractions
    if is_strand_specific:
        data_pos = data[data['strand'] == '+'].reset_index(drop=True)
        data_neg = data[data['strand'] == '-'].reset_index(drop=True)
        data = calculate_adar_fractions(data_neg, data_pos, old_names, is_fractions)
    else:
        data_pos = data[data['reference'] == 'A'].reset_index(drop=True)
        data_neg = data[data['reference'] == 'T'].reset_index(drop=True)
        data = calculate_adar_fractions(data_neg, data_pos, old_names, is_fractions)

    save_and_log(data, path, tag)


def add_apobec_fractions(path, data_name, is_strand_specific, is_fractions, file_name_results=None):
    # read in data
    if not file_name_results:
        file_name_results = path + data_name + '_APOBEC_denoised.tsv'
        tag = "APOBEC"
    else:
        tag = file_name_results.split('/')[-1].split('.')[0] + '_APOBEC'
    data = pd.read_table(file_name_results)
    old_names = [name for name in data.columns]

    # add fractions
    if is_strand_specific:
        data_pos = data[data['strand'] == '+'].reset_index(drop=True)
        data_neg = data[data['strand'] == '-'].reset_index(drop=True)
        data = calculate_apobec_fractions(data_neg, data_pos, old_names, is_fractions)
    else:
        data_pos = data[data['reference'] == 'C'].reset_index(drop=True)
        data_neg = data[data['reference'] == 'G'].reset_index(drop=True)
        data = calculate_apobec_fractions(data_neg, data_pos, old_names, is_fractions)

    save_and_log(data, path, tag)


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


def old_fraction(x, type_i, A_i, C_i, G_i, T_i, is_fraction):
    type = x[type_i]
    A = x[A_i]
    C = x[C_i]
    G = x[G_i]
    T = x[T_i]
    tmp_dict = {'A': A, 'C': C, 'G': G, 'T': T}
    if is_fraction:
        return tmp_dict[type[1]]
    return tmp_dict[type[1]] / float(tmp_dict[type[1]] + tmp_dict[type[0]])


def clean_fraction(x, type_i, Apred_i, Cpred_i, Gpred_i, Tpred_i, is_fraction):
    type = x[type_i]
    A = x[Apred_i]
    C = x[Cpred_i]
    G = x[Gpred_i]
    T = x[Tpred_i]
    tmp_dict = {'A': A, 'C': C, 'G': G, 'T': T}
    coverage = float(tmp_dict[type[1]] + tmp_dict[type[0]])
    if coverage == 0:
        return None
    if is_fraction:
        return tmp_dict[type[1]]
    return tmp_dict[type[1]] / coverage


# replace nucleotides with complimentary
def invert(x, type_i, strand_i):
    strand = x[strand_i]
    type = x[type_i]
    pairs = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    if strand == '-':
        return pairs[type[0]] + pairs[type[1]]
    return type


def add_mixed_fractions(path, tag, file_name, is_strand_specific, is_fractions):
    # read data
    data = pd.read_table(file_name)
    old_names = [name for name in data.columns]

    # find all necessary columns
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

    # choose type and calculate fractions
    data = pd.concat([data, data.apply(lambda x: choose_type(x, ref_i, A_i, C_i, G_i, T_i), axis=1)],
                     axis=1, ignore_index=True)
    type_i = data.shape[1] - 1
    data = pd.concat([data, data.apply(lambda x: old_fraction(x, type_i, A_i, C_i, G_i, T_i, is_fractions), axis=1)],
                     axis=1, ignore_index=True)
    data = pd.concat([data, data.apply(lambda x: clean_fraction(x, type_i, Apred_i,
                                                                Cpred_i, Gpred_i, Tpred_i, is_fractions), axis=1)],
                     axis=1, ignore_index=True)
    if is_strand_specific:
        data = pd.concat([data, data.apply(lambda x: invert(x, type_i, strand_i), axis=1)], axis=1, ignore_index=True)
        data = data.drop(type_i, axis=1)
        old_names.extend(['fraction_ini', 'fraction_clean', 'type'])
    else:
        old_names.extend(['type', 'fraction_ini', 'fraction_clean'])

    data.columns = old_names
    save_and_log(data, path, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-d', '--dataDir', help='directory with files', required=True)
    parser.add_argument('-u', '--usual', help='process standard datasets: ADAR, APOBEC, SNP', action='store_true')
    parser.add_argument('-n', '--nonStrandSpecific', help="don't use strand info", action='store_true')
    parser.add_argument('-a', '--customFile', help='predict on custom file only', default='')
    parser.add_argument('-t', '--type', help='specify know editing type for your custom file: ADAR or APOBEC',
                        required=False, default='')
    parser.add_argument('-f', '--fractions', help='fractions were used instead of absolute values', required=False,
                        action='store_true')

    args = parser.parse_args()

    path = args.dataDir
    if not path.endswith('/'):
        path += '/'
    tag = args.dataName
    is_standard = args.usual
    is_strand_specific = (not args.nonStrandSpecific)
    is_fractions = args.fractions
    custom_file_name = args.customFile
    is_custom = (custom_file_name != '')
    set_type = args.type

    if set_type not in ['', 'ADAR', 'APOBEC']:
        print "Warning! Incorrect editing type provided.\nIt will be ignored.\n"
        set_type = ''

    # TODO: parameters that can't be used together

    if is_standard:
        add_adar_fractions(path, tag, is_strand_specific, is_fractions)
        add_apobec_fractions(path, tag, is_strand_specific, is_fractions)
        add_mixed_fractions(path, 'SNP', path + tag + '_SNP_denoised.tsv', is_strand_specific, is_fractions)

    # TODO: test it (naming especially)
    if is_custom:
        tag = custom_file_name.split('/')[-1].split('.')[0]
        if set_type == '':
            add_mixed_fractions(path, tag, custom_file_name, is_strand_specific, is_fractions)
        elif set_type == 'ADAR':
            add_adar_fractions(path, tag, is_strand_specific, is_fractions, custom_file_name)
        elif set_type == 'APOBEC':
            add_apobec_fractions(path, tag, is_strand_specific, is_fractions, custom_file_name)


if __name__ == "__main__":
    main()
