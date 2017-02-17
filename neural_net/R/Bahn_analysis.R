library(readr)
library(ggplot2)
library(dplyr)

data_dir <- "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/results/Bahn/"

load_data <- function(data_dir, target_name) {
  sample_name <- "GSE28040.nsa"
  file_name <- paste0(data_dir, sample_name, "/train_predict_50nodes_400epochs/", target_name, "_fractions.tsv")
  data <- read_tsv(file_name)
  data <- mutate(data, 
                 clean_coverage = A_pred + C_pred + T_pred + G_pred,
                 dataset = sample_name)
  
  all_datasets <- data
  
  sample_name <- "GSE28040.nsb"
  file_name <- paste0(data_dir, sample_name, "/train_predict_50nodes_400epochs/", target_name, "_fractions.tsv")
  data <- read_tsv(file_name)
  data <- mutate(data, 
                 clean_coverage = A_pred + C_pred + T_pred + G_pred,
                 dataset = sample_name)
  
  all_datasets <- bind_rows(all_datasets, data)
  
  sample_name <- "GSE28040.sia"
  file_name <- paste0(data_dir, sample_name, "/train_predict_50nodes_400epochs/", target_name, "_fractions.tsv")
  data <- read_tsv(file_name)
  data <- mutate(data, 
                 clean_coverage = A_pred + C_pred + T_pred + G_pred,
                 dataset = sample_name)
  
  all_datasets <- bind_rows(all_datasets, data)
  
  sample_name <- "GSE28040.sib"
  file_name <- paste0(data_dir, sample_name, "/train_predict_50nodes_400epochs/", target_name, "_fractions.tsv")
  data <- read_tsv(file_name)
  data <- mutate(data, 
                 clean_coverage = A_pred + C_pred + T_pred + G_pred,
                 dataset = sample_name)
  
  all_datasets <- bind_rows(all_datasets, data)
  rm(data)
  return(all_datasets)
}

# ADAR
all_datasets <- load_data(data_dir, "ADAR")

# number of initial sites
ini_n_nsa <- nrow(filter(all_datasets, dataset == "GSE28040.nsa"))
ini_n_nsb <- nrow(filter(all_datasets, dataset == "GSE28040.nsb"))
ini_n_sia <- nrow(filter(all_datasets, dataset == "GSE28040.sia"))
ini_n_sib <- nrow(filter(all_datasets, dataset == "GSE28040.sib"))
ini_n_nsa
ini_n_nsb
ini_n_sia
ini_n_sib

# NA in new fractions
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsa"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsb"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sia"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sib"))
all_datasets %>%
  filter(coverage < 750) %>%
  ggplot() +
  geom_density(aes(coverage, fill = is.na(fraction_clean)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~ dataset, ncol = 2, scales = "free") +
  labs(title = 'Coverage "threshold"')

all_points_with_fractions <- filter(all_datasets, !is.na(fraction_clean))
ggplot(all_points_with_fractions) +
  geom_point(aes(fraction_ini, fraction_clean, colour = in_dbsnp)) +
  theme_bw() +
  scale_colour_manual(values = c("black", "yellow")) +
  labs(title = paste0(target_name, " fractions")) +
  facet_wrap(~ dataset, ncol = 2) 

nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsa")) * 100 / ini_n_nsa
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsb")) * 100 / ini_n_nsb
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sia")) * 100 / ini_n_sia
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sib")) * 100 / ini_n_sib

ggplot(all_points_with_fractions) +
  geom_density(aes(fraction_ini, fill = (fraction_clean > 0)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Initial fractions density") +
  facet_wrap(~ dataset, ncol = 2, scales = "free")

# RADAR
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.nsa"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.nsa" & in_dbsnp == "True"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.nsb"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.nsb" & in_dbsnp == "True"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.sia"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.sia" & in_dbsnp == "True"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.sib"))
nrow(filter(all_points_with_fractions, fraction_clean > 0 & in_RADAR == "True" & dataset == "GSE28040.sib" & in_dbsnp == "True"))

ggplot(filter(all_points_with_fractions, in_RADAR == "True")) +
  geom_point(aes(fraction_ini, fraction_clean, colour = in_dbsnp)) +
  theme_bw() +
  scale_colour_manual(values = c("black", "yellow")) +
  labs(title = "ADAR fractions of sites from RADAR") +
  facet_wrap(~ dataset, ncol = 2) 

# coverage
ggplot(all_datasets) +
  geom_poin(aes(coverage, clean_coverage)) +
  theme_bw() +
  labs(title = "Coverage transformation") +
  facet_wrap(~ dataset, ncol = 2)

# Bahn list
bahn_list <- filter(all_datasets, in_Bahn == "True")

ggplot(bahn_list) +
  geom_density(aes(fraction_ini)) +
  theme_bw() +
  facet_wrap(~dataset, ncol = 2)

ggplot(bahn_list) +
  geom_density(aes(coverage)) +
  theme_bw() +
  facet_wrap(~dataset, ncol = 2)

nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.nsa"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.nsa" & in_dbsnp == "True"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.nsb"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.nsb" & in_dbsnp == "True"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.sia"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.sia" & in_dbsnp == "True"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.sib"))
nrow(filter(all_datasets, fraction_clean > 0 & in_Bahn == "True" & dataset == "GSE28040.sib" & in_dbsnp == "True"))

ggplot(filter(all_datasets, in_Bahn == "True")) +
  geom_point(aes(fraction_ini, fraction_clean, colour = in_dbsnp)) +
  theme_bw() +
  scale_colour_manual(values = c("black", "yellow")) +
  labs(title = "ADAR fractions of sites from Bahn list") +
  facet_wrap(~ dataset, ncol = 2) 

# APOBEC
all_datasets <- load_data(data_dir, "APOBEC")

# number of initial sites
ini_n_nsa <- nrow(filter(all_datasets, dataset == "GSE28040.nsa"))
ini_n_nsb <- nrow(filter(all_datasets, dataset == "GSE28040.nsb"))
ini_n_sia <- nrow(filter(all_datasets, dataset == "GSE28040.sia"))
ini_n_sib <- nrow(filter(all_datasets, dataset == "GSE28040.sib"))
ini_n_nsa
ini_n_nsb
ini_n_sia
ini_n_sib

# NA in new fractions
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsa"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsb"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sia"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sib"))
all_datasets %>%
  filter(coverage < 750) %>%
  ggplot() +
  geom_density(aes(coverage, fill = is.na(fraction_clean)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~ dataset, ncol = 2, scales = "free") +
  labs(title = 'Coverage "threshold"')

all_points_with_fractions <- filter(all_datasets, !is.na(fraction_clean))
ggplot(all_points_with_fractions) +
  geom_point(aes(fraction_ini, fraction_clean, colour = in_dbsnp)) +
  theme_bw() +
  scale_colour_manual(values = c("black", "yellow")) +
  labs(title = "APOBEC fractions") +
  facet_wrap(~ dataset, ncol = 2) 

nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsa")) * 100 / ini_n_nsa
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsb")) * 100 / ini_n_nsb
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sia")) * 100 / ini_n_sia
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sib")) * 100 / ini_n_sib

ggplot(all_points_with_fractions) +
  geom_density(aes(fraction_ini, fill = (fraction_clean > 0)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Initial fractions density") +
  facet_wrap(~ dataset, ncol = 2, scales = "free")

# SNP
all_datasets <- load_data(data_dir, "SNP")

# number of initial sites
ini_n_nsa <- nrow(filter(all_datasets, dataset == "GSE28040.nsa"))
ini_n_nsb <- nrow(filter(all_datasets, dataset == "GSE28040.nsb"))
ini_n_sia <- nrow(filter(all_datasets, dataset == "GSE28040.sia"))
ini_n_sib <- nrow(filter(all_datasets, dataset == "GSE28040.sib"))
ini_n_nsa
ini_n_nsb
ini_n_sia
ini_n_sib

ggplot(all_datasets) +
  geom_histogram(aes(coverage), bins = 100) +
  theme_bw() +
  labs(title = "Coverage distribution") +
  facet_wrap(~ dataset, ncol = 2)

ggplot(filter(all_datasets, coverage < 500)) +
  geom_histogram(aes(coverage), bins = 100) +
  theme_bw() +
  labs(title = "Coverage distribution") +
  facet_wrap(~ dataset, ncol = 2)

# NA in new fractions
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsa"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.nsb"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sia"))
nrow(filter(all_datasets, is.na(fraction_clean) & dataset == "GSE28040.sib"))
all_datasets %>%
  filter(coverage < 750) %>%
  ggplot() +
  geom_density(aes(coverage, fill = is.na(fraction_clean)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  facet_wrap(~ dataset, ncol = 2, scales = "free") +
  labs(title = 'Coverage "threshold"')

all_points_with_fractions <- filter(all_datasets, !is.na(fraction_clean))
ggplot(all_points_with_fractions) +
  geom_point(aes(fraction_ini, fraction_clean)) +
  theme_bw() +
  labs(title = "APOBEC fractions") +
  facet_wrap(~ dataset, ncol = 2) 

nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsa")) * 100 / ini_n_nsa
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.nsb")) * 100 / ini_n_nsb
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sia")) * 100 / ini_n_sia
nrow(filter(all_points_with_fractions, fraction_clean > 0 & dataset == "GSE28040.sib")) * 100 / ini_n_sib

ggplot(all_points_with_fractions) +
  geom_density(aes(fraction_ini, fill = (fraction_clean > 0)), alpha = 0.5) +
  theme_bw() +
  scale_fill_brewer(palette = "Set1") +
  labs(title = "Initial fractions density") +
  facet_wrap(~ dataset, ncol = 2, scales = "free")

