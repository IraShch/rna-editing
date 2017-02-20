#!/usr/bin/python
__author__ = 'Ira'

# Add three columns to bam_reader output:
# possible ADAR editing site?
# possible APOBEC editing site?
# position in dbSNP?

import argparse


def make_sets_nonss(file_name):
    chr_dict = {}
    is_first = True
    with open(file_name) as inp:
        for line in inp:
            # skip header
            if is_first:
                is_first = False
                continue

            l = line.strip().split('\t')

            # add position to dictionary
            chrom = l[0]
            position = int(l[1])
            if chrom not in chr_dict:
                chr_dict[chrom] = {}
            chr_dict[chrom][position] = line.strip()
    return chr_dict


def make_sets(file_name):
    chr_dict = {}
    is_first = True
    with open(file_name) as inp:
        for line in inp:
            # skip header
            if is_first:
                is_first = False
                continue

            l = line.strip().split('\t')

            # add position to dictionary
            chrom = l[0]
            if chrom not in chr_dict:
                chr_dict[chrom] = {}
            if int(l[1]) not in chr_dict[chrom]:
                chr_dict[chrom][int(l[1])] = {}
            chr_dict[chrom][int(l[1])][l[2]] = line.strip()
    return chr_dict


def add_info(filename, basename, outname, is_clean):
    # read all positions from file
    positions = make_sets(filename)

    # mark potential editing sites (ADAR editing, APOBEC editing)
    for chrom in positions:
        for pos in positions[chrom]:
            for strand in positions[chrom][pos]:
                current_line = positions[chrom][pos][strand].split('\t')
                reference_base = current_line[3]
                strand = current_line[2]
                G_count = float(current_line[6])
                C_count = float(current_line[5])
                A_count = float(current_line[4])
                T_count = float(current_line[7])
                # possible ADAR editing site?
                if (strand == "+" and reference_base == "A" and G_count > 0) or \
                        (strand == "-" and reference_base == "T" and C_count > 0):
                    positions[chrom][pos][strand] += '\tTRUE'
                else:
                    positions[chrom][pos][strand] += '\tFALSE'
                # possible APOBEC editing site?
                if (strand == "+" and reference_base == "C" and T_count > 0) or \
                        (strand == "-" and reference_base == "G" and A_count > 0):
                    positions[chrom][pos][strand] += '\tTRUE'
                else:
                    positions[chrom][pos][strand] += '\tFALSE'

    # iterate over dbSNP
    is_first = True
    with open(basename) as dbSNP:
        delete_chr = False
        add_chr = False
        for line in dbSNP:
            # skip header
            if line[0] == '#':
                continue

            l = line.strip().split('\t')

            # check names consistency in the first line
            if is_first:
                is_first = False
                if positions.keys()[0].startswith('chr') and not l[0].startswith('chr'):
                    add_chr = True
                    print "add chr to dbSNP chr names"
                if not positions.keys()[0].startswith('chr') and l[0].startswith('chr'):
                    delete_chr = True

            # change chr if necessary
            chrom = l[0]
            if delete_chr:
                if chrom == 'chrM':
                    chrom = 'MT'
                else:
                    chrom = chrom[3:]
            if add_chr:
                if chrom == 'MT':
                    chrom = 'chrM'
                else:
                    chrom = 'chr' + chrom

            # check if intersection
            if chrom in positions and int(l[1]) in positions[chrom]:
                for strand in positions[chrom][int(l[1])]:
                    current_line = positions[chrom][int(l[1])][strand].split('\t')
                    if len(current_line) < 12:
                        if not is_clean:
                            positions[chrom][int(l[1])][strand] += '\tTRUE'
                        else:
                            counts = {'A': float(current_line[4]), 'C': float(current_line[5]),
                                      'G': float(current_line[6]), 'T': float(current_line[7])}
                            if current_line[3] == l[3] and counts[l[4]] > 0:
                                positions[chrom][int(l[1])][strand] += '\tTRUE'

    # add flags to non-snp positions
    for chrom in positions:
        for pos in positions[chrom]:
            for strand in positions[chrom][pos]:
                current_line = positions[chrom][pos][strand].split('\t')
                if len(current_line) < 12:
                    positions[chrom][pos][strand] += '\tFALSE'

    # write new lines
    is_first = True
    with open(filename) as inp:
        with open(outname, 'w') as out:
            for line in inp:
                if is_first:
                    out.write(line.strip() + '\tcan_be_ADAR_editing\tcan_be_APOBEC_editing\tin_dbsnp\n')
                    is_first = False
                    continue

                l = line.strip().split('\t')
                out.write(positions[l[0]][int(l[1])][l[2]] + '\n')


def add_info_nonss(filename, basename, outname, is_clean):
    # read all positions from file
    positions = make_sets_nonss(filename)

    # mark potential editing sites (ADAR editing, APOBEC editing)
    for chrom in positions:
        for pos in positions[chrom]:
            current_line = positions[chrom][pos].split('\t')
            reference_base = current_line[3]
            strand = current_line[2]
            G_count = float(current_line[6])
            C_count = float(current_line[5])
            A_count = float(current_line[4])
            T_count = float(current_line[7])
            # possible ADAR editing site?
            if (reference_base == "A" and G_count > 0) or \
                    (reference_base == "T" and C_count > 0):
                positions[chrom][pos] += '\tTRUE'
            else:
                positions[chrom][pos] += '\tFALSE'
            # possible APOBEC editing site?
            if (reference_base == "C" and T_count > 0) or \
                    (reference_base == "G" and A_count > 0):
                positions[chrom][pos] += '\tTRUE'
            else:
                positions[chrom][pos] += '\tFALSE'

    # iterate over dbSNP
    is_first = True
    with open(basename) as dbSNP:
        delete_chr = False
        add_chr = False
        for line in dbSNP:
            # skip header
            if line[0] == '#':
                continue

            l = line.strip().split('\t')

            # check names consistency in the first line
            if is_first:
                is_first = False
                if positions.keys()[0].startswith('chr') and not l[0].startswith('chr'):
                    add_chr = True
                    print "add chr to dbSNP chr names"
                if not positions.keys()[0].startswith('chr') and l[0].startswith('chr'):
                    delete_chr = True

            # change chr if necessary
            chrom = l[0]
            if delete_chr:
                if chrom == 'chrM':
                    chrom = 'MT'
                else:
                    chrom = chrom[3:]
            if add_chr:
                if chrom == 'MT':
                    chrom = 'chrM'
                else:
                    chrom = 'chr' + chrom

            # check if intersection
            if chrom in positions and int(l[1]) in positions[chrom]:
                current_line = positions[chrom][int(l[1])].split('\t')
                if len(current_line) < 12:
                    if not is_clean:
                        positions[chrom][int(l[1])] += '\tTRUE'
                    else:
                        counts = {'A': float(current_line[4]), 'C': float(current_line[5]),
                                  'G': float(current_line[6]), 'T': float(current_line[7])}
                        if current_line[3] == l[3] and counts[l[4]] > 0:
                            positions[chrom][int(l[1])] += '\tTRUE'

    # add flags to non-snp positions
    for chrom in positions:
        for pos in positions[chrom]:
            current_line = positions[chrom][pos].split('\t')
            if len(current_line) < 12:
                positions[chrom][pos] += '\tFALSE'

    # write new lines
    is_first = True
    with open(filename) as inp:
        with open(outname, 'w') as out:
            for line in inp:
                if is_first:
                    out.write(line.strip() + '\tcan_be_ADAR_editing\tcan_be_APOBEC_editing\tin_dbsnp\n')
                    is_first = False
                    continue

                l = line.strip().split('\t')
                out.write(positions[l[0]][int(l[1])] + '\n')


def main():
    parser = argparse.ArgumentParser()
    input_files = parser.add_argument_group('input')
    input_files.add_argument('-i', '--input', required=True,
                             help='input file: tab-delimited file, 1st column - chromosome, 2nd - position')
    input_files.add_argument('-r', '--reference', help='genome assembly version (37 or 38)',
                             choices=[37, 38], required=True, type=int)
    input_files.add_argument('-c', '--clean', help='if dbSNP contains only one type of change per position '
                                                   'and doesn\'t have indels',
                             action='store_true')
    input_files.add_argument('-s', '--strandSpecific', help='if the expreiment is strand specific', action='store_true')
    out_files = parser.add_argument_group('output')
    out_files.add_argument('-o', '--outname', help='name of output file', required=True)

    args = parser.parse_args()

    input_file = args.input
    output_file = args.outname
    is_clean = args.clean
    is_strand_specific = args.strandSpecific

    if args.reference == 37:
        if is_clean:
            basename = '/home/schukina/data/common/dbSNP37_clean.vcf'
        else:
            basename = '/home/schukina/data/common/dbSNP37.vcf'
    else:
        if is_clean:
            basename = '/home/schukina/data/common/dbSNP38_clean.vcf'
        else:
            basename = '/home/schukina/data/common/dbSNP38.vcf'

    if is_strand_specific:
        add_info(input_file, basename, output_file, is_clean)
    else:
        add_info_nonss(input_file, basename, output_file, is_clean)


if __name__ == "__main__":
    main()
