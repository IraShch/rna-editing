# in database file:
# 1 - chromosome
# 2 - position
# strand info if provided: "strand"

import argparse
import pandas as pd


def add_column(input_file, database_file, database_name, use_strand, output_file):
    data = pd.read_table(input_file)

    # define keys to intersect on
    keys = ['seqnames', 'pos']
    if use_strand:
        print "Check correctness!!!"
        keys.append('strand')

    # prepare database
    database = pd.read_table(database_file)
    old_database_names = [name for name in database.columns]
    old_database_names[0] = "seqnames"
    old_database_names[1] = "pos"
    database.columns = old_database_names
    database = database[keys]
    database = database.assign(in_database=pd.Series(True, index=database.index).values)

    # add column
    data = data.merge(database, on=keys, how='left')
    data['in_database'].fillna(False, inplace=True)
    data_names = [name for name in data.columns]
    column_name = 'in_{}'.format(database_name)
    data_names[-1] = column_name
    data.columns = data_names

    # count intersection
    print "Rows in data: {}".format(data.shape[0])
    print "Rows in database: {}".format(database.shape[0])
    print "Rows in intersection: {}".format(sum(data[column_name]))

    # save
    data.to_csv(output_file, sep='\t', index=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', help='Name of input file', required=True)
    parser.add_argument('-l', '--listFile', help='File with the list to intersect with', required=True)
    parser.add_argument('-n', '--listName', help='Name of the list', required=True)
    parser.add_argument('-o', '--outputFile', help='Output file', required=False, default='')
    parser.add_argument('-s', '--useStrand', help='Consider strand when intersecting', action='store_true')

    args = parser.parse_args()

    input_file = args.inputFile
    database_file = args.listFile
    database_name = args.listName
    if args.outputFile != '':
        output_file = args.outputFile
    else:
        output_file = input_file
    use_strand = args.useStrand

    add_column(input_file, database_file, database_name, use_strand, output_file)

if __name__ == "__main__":
    main()
