import pandas as pd

def check_duplicated_positions(x):
    duplicated_rows = x.duplicated(subset=['seqnames', 'pos', 'strand'], keep=False)
    print sum(duplicated_rows)
    print x[duplicated_rows]

directory_name = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/Bahn/'

initial_file_name = directory_name + 'GSE28040.nsb.badstrand.tsv'
ini_data = pd.read_table(initial_file_name)


check_duplicated_positions(ini_data)
print ini_data.shape
