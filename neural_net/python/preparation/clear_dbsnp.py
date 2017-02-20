#!/usr/bin/python

# filter dbSNP:
# removes multiallelic SNPs and indels

import argparse


def clean_dbsnp(input_name, output_name):
    is_first = True
    with open(input_name) as inp:
        with open(output_name, 'w') as out:
            for line in inp:
                # copy header
                if line.startswith('#'):
                    out.write(line)
                    continue

                l = line.strip().split()
                reference = l[3]
                alteration = l[4]

                if len(reference) > 1 or len(alteration) > 1:
                    continue
                if not is_first and l[0] == previous[0] and l[1] == previous[1]:
                    continue

                if is_first:
                    is_first = False
                out.write(line)
                previous = l


def main():
    parser = argparse.ArgumentParser()
    input_files = parser.add_argument_group('input')
    input_files.add_argument('-i', '--input', help='input vcf with initial dbSNP', required=True)
    out_files = parser.add_argument_group('output')
    out_files.add_argument('-o', '--output', help='output file name', required=True)

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output

    clean_dbsnp(input_file, output_file)


if __name__ == "__main__":
    main()
