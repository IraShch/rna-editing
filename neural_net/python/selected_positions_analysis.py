import pandas as pd
import argparse


def intersect_with_database(database_file, path, base_name):
    database = pd.read_table(database_file).iloc[:, 0:2]
    database.columns = ['seqnames', 'pos']
    file_name_results = path + 'ADAR_fractions.tsv'
    data = pd.read_table(file_name_results)
    intersection = pd.merge(data, database, how='inner', on=['seqnames', 'pos'])
    remaining = intersection[intersection['fraction_clean'] > 0].shape[0]

    #  save data
    output_file_name = path + base_name + '_fractions.tsv'
    intersection.to_csv(output_file_name, sep='\t', index=False)

    log_file_name = path + 'log_predict.txt'
    with open(log_file_name, 'a') as out:
        out.write('\n# {} sites\n'.format(base_name))
        out.write("Initial # sites: {}\n".format(data.shape[0]))
        out.write("# sites in database: {}\n".format(database.shape[0]))
        out.write("Intersection between initial set and database: {}\n".format(intersection.shape[0]))
        out.write("# sites from database remaining after cleaning: {} ({}%)\n".format(remaining,
                                                                                      100 * remaining / float(intersection.shape[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--baseFile', help='database file', required=True)
    parser.add_argument('-b', '--baseName', help='name of the database', required=True)
    parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)

    args = parser.parse_args()

    path = args.dataDir
    if not path.endswith('/'):
        path += '/'

    intersect_with_database(args.baseFile, path, args.baseName)


if __name__ == "__main__":
    main()




