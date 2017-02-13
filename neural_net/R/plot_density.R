library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(argparse)

source("~/Documents/Ira/soft/neural_net/rna-editing/neural_net/R/multiplot.R")

# parse command line arguments
parser <- ArgumentParser()
parser$add_argument('-s', '--setName', help = 'set name (ADAR/APOBEC/SNP)', required = TRUE)
parser$add_argument('-d', '--dataDir', help = 'path to files', required = TRUE)
args <- parser$parse_args()

path <- args$dataDir
if (!endsWith(path, "/")) {
  path <- paste0(path, "/")
}

file_name <- paste0(path, args$setName, "_fractions.tsv")

fractions <- read_tsv(file_name) %>%
  filter(!is.na(fraction_ini) & !is.na(fraction_clean) & fraction_clean > 0) %>%
  gather("step", "fraction", 14:15) %>%
  mutate(step = plyr::revalue(as.factor(step), c("fraction_ini"="noisy", "fraction_clean"="clean")))

both <- fractions %>%
  ggplot() +
  geom_density(aes(fraction, fill = step), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = 'Before/after')
clean <- fractions %>%
  filter(step == 'clean') %>%
  ggplot() +
  geom_density(aes(fraction, fill = step), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = 'Only clean distribution') +
  theme(legend.position="none")

out_name <- paste0(path, args$setName, "_density.pdf")
pdf(out_name, width = 10)
multiplot(both, clean, cols=2)
dev.off()
