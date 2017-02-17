import pandas as pd
import argparse


def replace_label(input_file, output_file):
    data = pd.read_table(input_file)
    data['strand'] = "*"
    data.to_csv(output_file, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser()
    input_files = parser.add_argument_group('input')
    input_files.add_argument('-i', '--input', required=True, help='input file to replace strand labels in')
    out_files = parser.add_argument_group('output')
    out_files.add_argument('-o', '--outname', help='name of output file', required=True)

    args = parser.parse_args()

    input_file = args.input
    output_file = args.outname

    replace_label(input_file, output_file)


if __name__ == "__main__":
    main()