import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--dataName', help='name of the dataset', required=True)
parser.add_argument('-d', '--dataDir', help='directory for data storage', required=True)
args = parser.parse_args()
data_dir = args.dataDir
data_name = args.dataName

all_sets = {}
intersection = {}
sizes = {}

for j in range(1, 6):
    file_name = data_dir + "{}_{}_adar{}_denoised.tsv".format(data_name, data_name, j)
    print "Read file {}".format(file_name)
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

            if (reference == 'A' and predicted_values['G'] > 0) or \
                    (reference == 'T' and predicted_values['C'] > 0):
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
    print "{}: {}%\n".format(i, percent)
print "Mean percent: {}\n".format(med / float(5))

for i in range(1, 6):
    for j in range(i+1, 6):
        intersection_size = 0
        for chr in all_sets[i]:
            intersection_size += len(all_sets[i][chr].intersection(all_sets[j][chr]))
        print "{}, {}: {}".format(i, j, intersection_size)
