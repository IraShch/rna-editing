#!/bin/bash

while getopts ":i:d:o:e:" opt
do
	case "$opt" in
		i)
			input_file="$OPTARG";;
		d)
			data_dir="$OPTARG";;
		o)
			result_dir="$OPTARG";;
		e)
			n_epoch="$OPTARG";;
	esac
done

TAG=$(basename $input_file)
TAG=${TAG%%.tsv}

python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/split_data.py -i $input_file -d $data_dir
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/train_test.py -i $TAG -d $data_dir -o $result_dir -e $n_epoch

# TODO: check if full path
out_dir=$result_dir$TAG"/train_test_50nodes_"$n_epoch"epochs/"
cd $out_dir
R -e "rmarkdown::render('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/train_test_report.Rmd', \
output_file = '"$out_dir$TAG"_report.html', params = list(model_name = '"$TAG"', dir = '"$out_dir"'))"
cd ..

out_dir=$result_dir$TAG"/train_predict_50nodes_"$n_epoch"epochs/"
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/train_predict.py -i $TAG -d $data_dir -o $result_dir -e $n_epoch
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/fraction_analysis.py -i $TAG -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s ADAR -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s APOBEC -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s SNP -d $out_dir
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/selected_positions_analysis.py -f ~/Documents/Ira/databases/RADAR_Human_AG_all_hg19_v2.txt -b RADAR -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s RADAR -d $out_dir