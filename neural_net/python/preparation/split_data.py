import pandas
import argparse

def split_data(file_name, data_dir, coverage_thr, include_coverage):
    file_name = file_name.split('/')[-1]
    data_name = file_name.strip('.tsv')
    if not data_dir.endswith('/'):
        data_dir += '/'
    input_file_name = data_dir + file_name
    log_file = data_dir + 'data_split_log.txt'

    # load initial file: bam_reader with 3 added columns
    data = pandas.read_table(input_file_name)
    data = data[data['coverage'] >= coverage_thr]

    # split data by type
    apobec = data[data['can_be_APOBEC_editing'] == True]
    adar = data[data['can_be_ADAR_editing'] == True]
    snp = data[data['in_dbsnp'] == True]
    noise = data[(data['can_be_APOBEC_editing'] == False) &
                 (data['can_be_ADAR_editing'] == False) &
                 (data['in_dbsnp'] == False)]

    with open(log_file, 'w') as out:
        out.write('Working directory: {}\n'.format(data_dir))
        out.write('Input file: {}\n'.format(file_name))
        out.write('Coverage threshold: {}\n\n'.format(coverage_thr))
        out.write('Data sizes.\n')
        out.write('Initial file: {}\n'.format(data.shape[0]))
        out.write('Noise: {}\n'.format(noise.shape[0]))
        out.write('ADAR: {}\n'.format(adar.shape[0]))
        out.write('APOBEC: {}\n'.format(apobec.shape[0]))
        out.write('SNP: {}'.format(snp.shape[0]))

    apobec.to_csv('{}{}_apobec.tsv'.format(data_dir, data_name), sep='\t', index=False)
    adar.to_csv('{}{}_adar.tsv'.format(data_dir, data_name), sep='\t', index=False)
    snp.to_csv('{}{}_snp.tsv'.format(data_dir, data_name), sep='\t', index=False)
    noise.to_csv('{}{}_noise.tsv'.format(data_dir, data_name), sep='\t', index=False)

    # create X and y for learning
    if include_coverage:
        X = pandas.DataFrame(noise, columns=['A', 'C', 'G', 'T', 'coverage'])
    else:
        X = pandas.DataFrame(noise, columns=['A', 'C', 'G', 'T'])
    y = pandas.DataFrame(noise, columns=['A', 'C', 'G', 'T'])
    y['C'] *= (noise['reference'] == 'C')
    y['A'] *= (noise['reference'] == 'A')
    y['T'] *= (noise['reference'] == 'T')
    y['G'] *= (noise['reference'] == 'G')

    if include_coverage:
        X.to_csv('{}{}_noise_X_cov.tsv'.format(data_dir, data_name), sep='\t', index=False)
        y.to_csv('{}{}_noise_y_cov.tsv'.format(data_dir, data_name), sep='\t', index=False)
    else:
        X.to_csv('{}{}_noise_X.tsv'.format(data_dir, data_name), sep='\t', index=False)
        y.to_csv('{}{}_noise_y.tsv'.format(data_dir, data_name), sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', help='name of input file', required=True)
    parser.add_argument('-d', '--directory', help='directory for data storage', required=True)
    parser.add_argument('-c', '--coverage', help='coverage threshold', required=False)
    parser.add_argument('-v', '--includeCoverage', help='include coverage column into X', action='store_true')

    args = parser.parse_args()

    if args.coverage:
        coverage_thr = int(args.coverage)
    else:
        coverage_thr = 10
    if args.includeCoverage:
        include_coverage = True
    else:
        include_coverage = False

    split_data(args.inputFile, args.directory, coverage_thr, include_coverage)


if __name__ == "__main__":
    main()

