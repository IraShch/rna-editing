#!/bin/bash

while getopts ":i:d:o:" opt
do
	case "$opt" in
		i)
			input_file="$OPTARG";;
		d)
			data_dir="$OPTARG";;
		o)
			result_dir="$OPTARG";;
	esac
done

TAG=$(basename $input_file)
TAG=${TAG%%.tsv}

python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/split_data.py -i $input_file -d $data_dir
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/train_test.py -i $TAG -d $data_dir -o $result_dir

# TODO: check if full path
out_dir=$result_dir$TAG/train_test_50nodes_400epochs/
cd $out_dir
R -e "rmarkdown::render('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/train_test_report.Rmd', \
output_file = '"$out_dir$TAG"_report.html', params = list(model_name = '"$TAG"', dir = '"$out_dir"'))"
cd ..

python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/train_predict.py -i $TAG -d $data_dir -o $result_dir
