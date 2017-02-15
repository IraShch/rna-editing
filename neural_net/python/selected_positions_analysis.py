import pandas as pd
import argparse


def custom_intersect_with_database(database_file, path, base_name, data_file):
    database = pd.read_table(database_file).iloc[:, 0:2]
    database.columns = ['seqnames', 'pos']
    data = pd.read_table(data_file)
    intersection = pd.merge(data, database, how='inner', on=['seqnames', 'pos'])
    remaining = intersection[intersection['fraction_clean'] > 0].shape[0]

    #  save data
    tag = data_file.split('/')[-1].split('.')[0]
    output_file_name = path + tag + '_' + base_name + '_fractions.tsv'
    intersection.to_csv(output_file_name, sep='\t', index=False)

    print('\n# {} sites\n'.format(base_name))
    print("Initial # sites: {}\n".format(data.shape[0]))
    print("# sites in database: {}\n".format(database.shape[0]))
    print("Intersection between initial set and database: {}\n".format(intersection.shape[0]))
    print("# sites from database remaining after cleaning: {} ({}%)\n".format(remaining,
                                                                                  100 * remaining / float(
                                                                                      intersection.shape[0])))
        


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
    parser.add_argument('-c', '--customFile', help='predict on custom file only', default='')

    args = parser.parse_args()

    path = args.dataDir
    if not path.endswith('/'):
        path += '/'
    is_custom = (args.customFile != '')

    if not is_custom:
        intersect_with_database(args.baseFile, path, args.baseName)
    else:
        custom_intersect_with_database(args.baseFile, path, args.baseName, args.customFile)


if __name__ == "__main__":
    main()




