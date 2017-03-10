#!/bin/bash

RADAR=""

while getopts ":i:d:o:e:r" opt
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
		r)
			RADAR=true;;
	esac
done

TAG=$(basename $input_file)
TAG=${TAG%%.tsv}

python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/preparation/split_data.py -i $input_file -d $data_dir
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/main/train_test.py -i $TAG -d $data_dir -o $result_dir -e $n_epoch

# TODO: check if full path
out_dir=$result_dir$TAG"/train_test_50nodes_"$n_epoch"epochs/"
cd $out_dir
R -e "rmarkdown::render('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/train_test_report.Rmd', \
output_file = '"$out_dir$TAG"_report.html', params = list(model_name = '"$TAG"', dir = '"$out_dir"'))"
cd ..

out_dir=$result_dir$TAG"/train_predict_50nodes_"$n_epoch"epochs/"
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/main/train_predict.py -i $TAG -d $data_dir -o $result_dir -e $n_epoch
python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/fraction_analysis.py -i $TAG -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s ADAR -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s APOBEC -d $out_dir
Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s SNP -d $out_dir

if [ "$RADAR" ]; then
    python /Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/python/selected_positions_analysis.py -f ~/Documents/Ira/databases/RADAR_Human_AG_all_hg19_v2.txt -b RADAR -d $out_dir
	Rscript ~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/plot_density.R -s RADAR -d $out_dir
fi


R -e "rmarkdown::render('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/train_test_report.Rmd', output_file = '~/Documents/Ira/soft/neural_net/results/Bahn/GSE28040.nsa/train_test_150nodes_400epochs_1coverage_1identical/GSE28040.nsa_report.html', params = list(model_name = 'GSE28040.nsa', dir = '~/Documents/Ira/soft/neural_net/results/Bahn/GSE28040.nsa/train_test_150nodes_400epochs_1coverage_1identical/'))"


R -e "rmarkdown::render('/Users/bioinformaticshub/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/train_test_report.Rmd', output_file = '/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/low_coverage_1000/GSE28040.sib.low/train_test_150nodes_400epochs_1coverage_1identical/GSE28040.sib.low_report.html', params = list(model_name = 'GSE28040.sib.low', dir = '~/Documents/Ira/soft/neural_net/results/Bahn/low_coverage_1000/GSE28040.sib.low/train_test_150nodes_400epochs_1coverage_1identical/'))"