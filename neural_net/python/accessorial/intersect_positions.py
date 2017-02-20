import copy

data_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/stability_test/" \
           "GSE28040.nsa/train_predict_50nodes_600epochs/"

all_sets = {}
intersection = {}
sizes = {}

for j in range(1, 6):
    file_name = data_dir + "GSE28040.nsa_GSE28040{}_denoised.tsv".format(j)
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

            if (reference == 'A' and predicted_values['G'] > 0 and predicted_values['A'] > 0) or \
                    (reference == 'T' and predicted_values['C'] > 0 and predicted_values['T'] > 0):
                all_sets[j][chr].add(pos)
                sizes[j] += 1

intersection = copy.deepcopy(all_sets[1])

for j in range(2, 6):
    for chr in intersection:
        intersection[chr].intersection_update(all_sets[j][chr])

sum_intersection = 0
for chr in intersection:
    sum_intersection += len(intersection[chr])
print "Total intersection: {}".format(sum_intersection)
print sizes

for i in range(1, 6):
    for j in range(i+1, 6):
        intersection_size = 0
        for chr in all_sets[i]:
            intersection_size += len(all_sets[i][chr].intersection(all_sets[j][chr]))
        print "{}, {}: {}".format(i, j, intersection_size)

# out_name = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/stability_test/" \
#            "GSE28040.nsa/100_600.tsv"
# with open(out_name, 'w') as out:
#     for chr in intersection:
#         for pos in intersection[chr]:
#             out.write('\t'.join([chr, pos]) + '\n')


print '\nIntersection of intersections\n'
data_dir = "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/stability_test/" \
           "GSE28040.nsa/"

all_sets = {}
intersection = {}
sizes = {}

for j in [200, 400, 600]:
    file_name = data_dir + "{}.tsv".format(j)
    all_sets[j] = {}
    sizes[j] = 0

    with open(file_name) as inp:
        for line in inp:
            l = line.strip().split('\t')
            chr = l[0]
            pos = l[1]

            if chr not in all_sets[j]:
                all_sets[j][chr] = set()
            all_sets[j][chr].add(pos)
            sizes[j] += 1


intersection = copy.deepcopy(all_sets[200])

for j in [400, 600]:
    for chr in intersection:
        intersection[chr].intersection_update(all_sets[j][chr])

sum_intersection = 0
for chr in intersection:
    sum_intersection += len(intersection[chr])
print "Total intersection: {}".format(sum_intersection)
print sizes

for i in [200, 400, 600]:
    for j in [400, 600]:
        intersection_size = 0
        for chr in all_sets[i]:
            intersection_size += len(all_sets[i][chr].intersection(all_sets[j][chr]))
        print "{}, {}: {}".format(i, j, intersection_size)