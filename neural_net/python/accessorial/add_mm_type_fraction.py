__author__ = 'Ira'

import argparse


def choose_type(reference, A, C, G, T):
    if reference == 'A':
        (non_ref_count, ind) = max((val, ind) for ind, val in enumerate([C, G, T]))
        return 'A' + ['C', 'G', 'T'][ind], non_ref_count / (A + non_ref_count)
    elif reference == 'C':
        (non_ref_count, ind) = max((val, ind) for ind, val in enumerate([A, G, T]))
        return 'C' + ['A', 'G', 'T'][ind], non_ref_count / (C + non_ref_count)
    elif reference == 'G':
        (non_ref_count, ind) = max((val, ind) for ind, val in enumerate([C, A, T]))
        return 'G' + ['C', 'A', 'T'][ind], non_ref_count / (G + non_ref_count)
    elif reference == 'T':
        (non_ref_count, ind) = max((val, ind) for ind, val in enumerate([C, G, A]))
        return 'T' + ['C', 'G', 'A'][ind], non_ref_count / (T + non_ref_count)


def characterise_mismatches(inp_file_name, data_name, output_dir):
    is_first = True
    out_name = output_dir + data_name + '_mm_fractions.tsv'
    with open(inp_file_name) as inp:
        with open(out_name, 'w') as out:
            for line in inp:
                l = line.strip().split()

                # parse the header
                if is_first:
                    reference_ind = l.index('reference')
                    A_index = l.index('A')
                    C_index = l.index('C')
                    G_index = l.index('G')
                    T_index = l.index('T')
                    dbsnp_index = l.index('in_dbsnp')
                    l.extend(['type', 'fraction'])
                    out.write('\t'.join(l) + '\n')
                    is_first = False
                    continue

                if l[dbsnp_index] == 'TRUE':
                    continue

                # print l
                type, fraction = choose_type(l[reference_ind], float(l[A_index]), float(l[C_index]), float(l[G_index]),
                                             float(l[T_index]))
                l.extend([type, str(fraction)])
                out.write('\t'.join(l) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', help='name of input file', required=True)
    parser.add_argument('-d', '--dataName', help='name of the dataset', required=True)
    parser.add_argument('-o', '--outputDir', help='directory to save the results', required=True)

    args = parser.parse_args()

    out_dir = args.outputDir
    if not out_dir.endswith('/'):
        out_dir += '/'

    characterise_mismatches(args.inputFile, args.dataName, out_dir)


if __name__ == "__main__":
    main()
