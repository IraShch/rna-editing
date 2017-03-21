import copy
import argparse
import pandas as pd


def calculate_threshold(file_name):
    data = pd.read_table(file_name)
    plus = data[data['reference'] == 'A']
    minus = data[data['reference'] == 'T']
    plus['noise'] = plus['T'] + plus['C']
    minus['noise'] = minus['A'] + minus['G']
    return (plus['noise'].mean() + minus['noise'].mean()) / float(2)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
parser.add_argument('-n', '--noise', help='use threshold calculated as average noise', action='store_true')
args = parser.parse_args()
data_dir = args.dataDir
data_name = args.dataName
use_threshold = args.noise

all_sets = {}
intersection = {}
sizes = {}

number_of_files = 5

for j in range(1, number_of_files + 1):
    file_name = data_dir + "{}_{}_adar{}_denoised.tsv".format(data_name, data_name, j)
    print "Read file {}".format(file_name)
    if use_threshold:
        threshold = calculate_threshold(file_name)
    else:
        threshold = 0
    all_sets[j] = {}
    sizes[j] = 0

    with open(file_name) as inp:
        is_first = True
        for line in inp:
            if is_first:
                is_first = False
                continue

            l = line.strip().split('\t')
            chr = l[0]
            pos = l[1]
            reference = l[3]
            predicted_values = {'A': float(l[12]), 'C': float(l[13]),
                                'G': float(l[14]), 'T': float(l[15])}

            if chr not in all_sets[j]:
                all_sets[j][chr] = set()

            if (reference == 'A' and predicted_values['G'] > threshold) or \
                    (reference == 'T' and predicted_values['C'] > threshold):
                all_sets[j][chr].add(pos)
                sizes[j] += 1

print "Intersect"
intersection = copy.deepcopy(all_sets[1])

for j in range(2, 6):
    for chr in intersection:
        intersection[chr].intersection_update(all_sets[j][chr])

sum_intersection = 0
for chr in intersection:
    sum_intersection += len(intersection[chr])
print "Total intersection: {}".format(sum_intersection)
print sizes
med = 0
for i in sizes:
    percent = sum_intersection * 100 / float(sizes[i])
    med += percent
    print "{}: {}%".format(i, percent)
print "Mean percent: {}%\n".format(med / float(5))

for i in range(1, 6):
    for j in range(i+1, 6):
        intersection_size = 0
        for chr in all_sets[i]:
            intersection_size += len(all_sets[i][chr].intersection(all_sets[j][chr]))
        print "{}, {}: {}".format(i, j, intersection_size)
