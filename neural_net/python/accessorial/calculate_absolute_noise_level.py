import argparse
import pandas as pd


def calc_abs_noise(x, ref_i, A_i, C_i, G_i, T_i):
    ref = x[ref_i]
    A = x[A_i]
    C = x[C_i]
    G = x[G_i]
    T = x[T_i]
    coverage = A + C + G + T
    if ref == "A":
        return coverage - A
    if ref == "T":
        return coverage - T
    if ref == "C":
        return coverage - C
    if ref == "G":
        return coverage - G


def calc_noise_level(input, output):
    data = pd.read_table(input)
    noise = data[(data['can_be_APOBEC_editing'] == False) &
                 (data['can_be_ADAR_editing'] == False) &
                 (data['in_dbsnp'] == False)]
    old_names = [name for name in noise.columns]

    ref_i = noise.columns.get_loc('reference')
    A_i = noise.columns.get_loc('A')
    C_i = noise.columns.get_loc('C')
    G_i = noise.columns.get_loc('G')
    T_i = noise.columns.get_loc('T')

    noise = pd.concat([noise, noise.apply(lambda x: calc_abs_noise(x, ref_i, A_i, C_i, G_i, T_i), axis=1)],
                      axis=1, ignore_index=True)
    old_names.extend(['noise_abs'])
    noise.columns = old_names

    noise.to_csv(output, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    input_files = parser.add_argument_group('input')
    input_files.add_argument('-i', '--input', required=True, help='input file')
    out_files = parser.add_argument_group('output')
    out_files.add_argument('-o', '--output', help='output file', required=True)

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    calc_noise_level(input_file, output_file)


if __name__ == "__main__":
    main()