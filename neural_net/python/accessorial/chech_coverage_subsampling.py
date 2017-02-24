import random
import pandas as pd

def add_identical(X, k, out_name):
    n_additional = int(X.shape[0] * k)
    X = X.assign(distribution=pd.Series("Initial" for _ in range(X.shape[0])).values)
    X = X.loc[:, ['coverage', 'distribution']]

    X_res = X

    for i in range(10):
        X_add = X.sample(n=n_additional, replace=True)
        X_add['distribution'] = "subsample_{}".format(i + 1)
        X_res = pd.concat([X_res, X_add], axis=0)

    X_res.to_csv(out_name, sep='\t', index=False)


# prepares data for learning
def prepare_training_data(data_dir, data_name, include_coverage, train_on_identical, out_name, k=1.0):
    # load initial noise set
    if include_coverage:
        noise_X_file_name = '{}{}_noise_X_cov.tsv'.format(data_dir, data_name)
    else:
        noise_X_file_name = '{}{}_noise_X.tsv'.format(data_dir, data_name)
        noise_y_file_name = '{}{}_noise_y.tsv'.format(data_dir, data_name)
    X_train = pd.read_table(noise_X_file_name)

    if train_on_identical:
        add_identical(X_train, k, out_name)


def main():
    data_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/data/Bahn/"
    out_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/coverage_subsample/"
    data_name = "GSE28040.sib"
    include_coverage = True
    train_on_identical = True
    percent_identical = 0.5

    # split into sets
    prepare_training_data(data_dir, data_name, include_coverage,
                          train_on_identical, out_dir + data_name + ".tsv", percent_identical)



if __name__ == "__main__":
    main()
