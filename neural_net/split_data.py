import pandas

data_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/BT20/"
input_file_name = data_dir + "BT20_Normoxia_1_info.tsv"

# load initial file: bam_reader with 3 added columns
data = pandas.read_table(input_file_name)

# split data by type
apobec = data[data['can_be_APOBEC_editing'] == True]
adar = data[data['can_be_ADAR_editing'] == True]
snp = data[data['in_dbsnp'] == True]
noise = data[(data['can_be_APOBEC_editing'] == False) &
             (data['can_be_ADAR_editing'] == False) &
             (data['in_dbsnp'] == False)]

print "APOBEC", apobec.shape
print "ADAR", adar.shape
print "SNP", snp.shape
print "noise", noise.shape

apobec.loc[:, 'seqnames':'coverage'].to_csv(data_dir + 'BT20_N1_apobec.tsv', sep='\t', index=False)
adar.loc[:, 'seqnames':'coverage'].to_csv(data_dir + 'BT20_N1_adar.tsv', sep='\t', index=False)
snp.loc[:, 'seqnames':'coverage'].to_csv(data_dir + 'BT20_N1_snp.tsv', sep='\t', index=False)
noise.loc[:, 'seqnames':'coverage'].to_csv(data_dir + 'BT20_N1_all_noise.tsv', sep='\t', index=False)

# transform noise df into X, y
noise_file_name = data_dir + "BT20_N1_all_noise.tsv"

# load initial noise set
noise = pandas.read_table(noise_file_name)

# filter by coverage
coverage_thr = 10
noise = noise[noise['coverage'] >= coverage_thr]
print "Initial dataframe shape:", noise.shape

# create X and y
X = pandas.DataFrame(noise, columns=['A', 'C', 'G', 'T'])
y = pandas.DataFrame(noise, columns=['A', 'C', 'G', 'T'])
y['C'] *= (noise['reference'] == 'C')
y['A'] *= (noise['reference'] == 'A')
y['T'] *= (noise['reference'] == 'T')
y['G'] *= (noise['reference'] == 'G')

X.to_csv(data_dir + 'BT20_N1_noise_X.tsv', sep='\t', index=False)
y.to_csv(data_dir + 'BT20_N1_noise_y.tsv', sep='\t', index=False)
