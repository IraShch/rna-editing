library(Rsamtools)
library(dplyr)
library(tidyr)
library(readr)
library(argparse)
library(tibble)
library(lubridate)
library(magrittr)
library(stringr)

is_mismatch <- function(reference, A, C, G, T){
  switch(reference,
         "A" = (C > 0) + (G > 0) + (T > 0) > 0,
         "T" = (A > 0) + (C > 0) + (G > 0) > 0,
         "G" = (A > 0) + (C > 0) + (T > 0) > 0,
         "C" = (A > 0) + (G > 0) + (T > 0) > 0,
         FALSE)
}

is_mismatch_vec <- Vectorize(is_mismatch)

# tries to define correct strand of the position via counting reads aligned to both strands
get_strand <- function(pos_first, current_chromosome_length, current_chromosome, bam_file) {
  eps <- 250
  left_range <- max(0, pos_first - eps)
  right_range <- min(current_chromosome_length, pos_first + eps)
  current_range <- GRanges(current_chromosome, IRanges(left_range, right_range))
  bam_flag <- scanBamFlag(isPaired = TRUE, 
                          isProperPair = TRUE, 
                          isSecondaryAlignment = FALSE,
                          isNotPassingQualityControls = FALSE,
                          isDuplicate = FALSE,
                          hasUnmappedMate = FALSE,
                          isFirstMateRead = TRUE)
  bam_params <- ScanBamParam(which = current_range, flag = bam_flag, what = c("strand"), mapqFilter = 30)
  first_area_info <- scanBam(bam_file, param = bam_params)[[1]][[1]]
  bam_flag <- scanBamFlag(isPaired = TRUE, 
                          isProperPair = TRUE, 
                          isSecondaryAlignment = FALSE,
                          isNotPassingQualityControls = FALSE,
                          isDuplicate = FALSE,
                          hasUnmappedMate = FALSE,
                          isSecondMateRead = TRUE)
  bam_params <- ScanBamParam(which = current_range, flag = bam_flag, what = c("strand"), mapqFilter = 30)
  second_area_info <- scanBam(bam_file, param = bam_params)[[1]][[1]]
  plus_read_count <- table(first_area_info)['+'] + table(second_area_info)['-']
  minus_read_count <- table(first_area_info)['-'] + table(second_area_info)['+']
  if (plus_read_count / minus_read_count > 2) {
    return("+")
  } else if (minus_read_count / plus_read_count > 2) {
    return("-")
  }
  return("*")
}

# actually get info from bam for given chromosome
fixed_chromosome_positions <- function(bam_file, current_chromosome, bam_index, pileup_params, fasta_file, 
                                       min_coverage, output_file, messy_file, ambiguous_file) {
  print(current_chromosome)
  
  # get raw pileup
  print("Get raw pileups from BAM")
  print(now())
  current_chromosome_length <- seqlengths(bam_file)[current_chromosome]
  current_range <- GRanges(current_chromosome, IRanges(1, current_chromosome_length))
  bam_flag <- scanBamFlag(isPaired = TRUE, 
                          isProperPair = TRUE, 
                          isSecondaryAlignment = FALSE,
                          isNotPassingQualityControls = FALSE,
                          isDuplicate = FALSE,
                          hasUnmappedMate = FALSE,
                          isFirstMateRead = TRUE)
  bam_params <- ScanBamParam(which = current_range, flag = bam_flag)
  pileup_table_first <- pileup(file = bam_file, 
                               index = bam_index, 
                               pileupParam = pileup_params, 
                               scanBamParam = bam_params)
  pileup_table_first$which_label <- NULL
  bam_flag <- scanBamFlag(isPaired = TRUE, 
                          isProperPair = TRUE, 
                          isSecondaryAlignment = FALSE,
                          isNotPassingQualityControls = FALSE,
                          isDuplicate = FALSE,
                          hasUnmappedMate = FALSE,
                          isSecondMateRead = TRUE)
  bam_params <- ScanBamParam(which = current_range, flag = bam_flag)
  pileup_table_second <- pileup(file = bam_file, 
                                index = bam_index, 
                                pileupParam = pileup_params, 
                                scanBamParam = bam_params)
  pileup_table_second$which_label <- NULL
  pileup_table_second <- pileup_table_second %>%
    mutate(strand = ifelse(strand == "*", "*", ifelse(strand == "+", "-", "+")))
  print(now())
  
  print("Combine two pileups")
  print(now())
  pileup_table <- bind_rows(pileup_table_first, pileup_table_second)
  rm(pileup_table_first, pileup_table_second)
  pileup_table <- pileup_table %>% 
    arrange(pos, strand, nucleotide) %>%
    mutate(is_next_same = (lead(pos) == pos & lead(strand) == strand & lead(nucleotide) == nucleotide)) %>%
    mutate(count = ifelse(!is.na(lag(is_next_same)), 
                          ifelse(lag(is_next_same), count + lag(count), count),
                          count)) %>%
    filter(!is_next_same) %>%
    select(-is_next_same)
  print(pryr::object_size(pileup_table))
  if (nrow(pileup_table) == 0) {return(NULL)}
  print(now())
  
  # get reference nucleotides
  print("Get reference nucleotides")
  print(now())
  positions <- GRanges(pileup_table$seqnames,
                       IRanges(start = pileup_table$pos, end = pileup_table$pos))
  reference_base <- getSeq(fasta_file, positions)
  pileup_table <- bind_cols(pileup_table, as.data.frame(reference_base)) %>% rename(reference = x)
  print(pryr::object_size(pileup_table))
  
  # spread rows by position
  print("Spread rows by position")
  print(now())
  pileup_table <- pileup_table %>% spread(key = nucleotide, value = count, fill = 0)
  print(pryr::object_size(pileup_table))
  
  # calculate coverage, fraction and filter by coverage
  print("Leave only mismatches, filter by coverage")
  print(now())
  pileup_table <- pileup_table %>% 
    mutate(is_mismatch = is_mismatch_vec(reference, A, C, G, T)) %>%
    filter(is_mismatch | 
             ((lead(pos) == pos) & lead(is_mismatch)) | 
             ((lag(pos) == pos) & lag(is_mismatch))) %>%
    mutate(coverage = A + G + T + C) %>%
    filter(coverage > min_coverage)
  pileup_table$is_mismatch <- NULL
  if (nrow(pileup_table) == 0) {return(NULL)}
  print(pryr::object_size(pileup_table))
  
  # choose strand
  print("Choose strand to use")
  print("Find paired positions, split data")
  print(now())
  pileup_table <- pileup_table %>% mutate(is_paired = (lead(pos) == pos | lag(pos) == pos))
  if (nrow(pileup_table) > 1) {
    pileup_table[1, "is_paired"] = (pileup_table[1, "pos"] == pileup_table[2, "pos"])
    pileup_table[nrow(pileup_table), "is_paired"] = (pileup_table[nrow(pileup_table), "pos"] == 
                                                       pileup_table[nrow(pileup_table) - 1, "pos"])
  } else {
    pileup_table[1, "is_paired"] = FALSE
  }
  pairs <- pileup_table %>% 
    filter(is_paired) %>%
    mutate(is_mess = 0)
  pileup_table <- pileup_table %>% filter(!is_paired)
  
  # 0 - don't know
  # 1 - correct
  # 2 - delete
  # try coverage ratio
  print("Try coverage ratio")
  print(now())
  pairs <- pairs %>% 
    mutate(is_mess = ifelse(row_number() %% 2 == 1, 
                            ifelse(coverage / lead(coverage) > 2, 
                                   1, 
                                   ifelse(lead(coverage) / coverage > 2, 2, 0)),
                            0)) %>%
    mutate(is_mess = ifelse(row_number() %% 2 == 0,
                            ifelse(lag(is_mess) == 1,
                                   2,
                                   ifelse(lag(is_mess) == 2, 1, 0)),
                            is_mess))
  pileup_table <- bind_rows(pileup_table, filter(pairs, is_mess == 1) %>% select(-is_mess))
  messy_positions <- filter(pairs, is_mess == 2) %>% select(-is_mess, -is_paired)
  pairs <- filter(pairs, is_mess == 0)
  
  # delete sites with low coverage
  print("Filter by coverage")
  print(now())
  coverage_threshold <- 10
  pairs <- pairs %>% 
    mutate(is_mess = ifelse(row_number() %% 2 == 1,
                            ifelse(coverage < coverage_threshold & lead(coverage) < coverage_threshold, 2, 0),
                            ifelse(coverage < coverage_threshold & lag(coverage) < coverage_threshold, 2, 0)))
  messy_positions <- bind_rows(messy_positions, filter(pairs, is_mess == 2) %>% select(-is_paired, -is_mess))
  pairs <- filter(pairs, is_mess == 0)
  
  # count reads
  # count reads
  print("Count reads around")
  print(now())
  if (nrow(pairs) > 0) {
    for (i in 1:nrow(pairs)) {
      if (i %% 2 == 0) {
        first_mate_flag <- pairs[i - 1, "is_mess"][[1]]
        if (first_mate_flag == 0) {
          pairs[i, "is_mess"] <- 0
        } else if (first_mate_flag == 1) {
          pairs[i, "is_mess"] <- 2
        } else {
          pairs[i, "is_mess"] <- 1
        }
      } else {
        correct_strand <- get_strand(pairs[i, "pos"][[1]], current_chromosome_length, current_chromosome, bam_file)
        if (correct_strand == "*") {
          pairs[i, "is_mess"] <- 0
        } else if (correct_strand == pairs[i, "strand"][[1]]) {
          pairs[i, "is_mess"] <- 1
        } else {
          pairs[i, "is_mess"] <- 2
        }
      }
    }
  }
  pileup_table <- bind_rows(pileup_table, filter(pairs, is_mess != 2) %>% select(-is_mess)) %>% 
    select(-is_paired) %>% 
    arrange(pos)
  messy_positions <- bind_rows(messy_positions, filter(pairs, is_mess == 2) %>% select(-is_mess, -is_paired)) %>%
    arrange(pos)
  pairs <- filter(pairs, is_mess == 0) %>% 
    select(-is_mess, -is_paired) %>%
    arrange(pos)
  
  # write tables
  print("Write files")
  print(now())
  write_tsv(pileup_table, output_file, append = TRUE)
  write_tsv(messy_positions, messy_file, append = TRUE)
  write_tsv(pairs, ambiguous_file, append = TRUE)
}

# wrapper function
get_positions_from_bam <- function(bam_file_name, bam_index, reference_fasta, output_file, max_depth, min_base_quality,
                                   min_mapq, min_nucleotide_depth, min_minor_allele_depth, min_coverage, use_noncanonical) {
  
  bam_file <- BamFile(bam_file_name, bam_index)
  fasta_file <- FaFile(file = reference_fasta)
  pileup_params <- PileupParam(max_depth = max_depth,
                               min_base_quality = min_base_quality, 
                               min_mapq = min_mapq, 
                               min_nucleotide_depth = min_nucleotide_depth,
                               min_minor_allele_depth = min_minor_allele_depth,
                               include_deletions = FALSE)
  chromosome_names <- names(seqlengths(bam_file))
  if (!use_noncanonical) {
    canonical_names <- c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                         "20", "21", "22", "X", "Y", "M", "MT", "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", 
                         "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", 
                         "chr20", "chr21", "chr22", "chrX", "chrY", "chrM", "chrMT")
    chromosome_names <- intersect(chromosome_names, canonical_names)
  }
  print("Chromosome names:")
  print(chromosome_names)
  
  # preprare output files
  path <- unlist(strsplit(output_file, "/", fixed = TRUE))
  n_path <- length(path)
  name_prefix <- unlist(strsplit(path[n_path], ".", fixed = TRUE))[1]
  messy_name <- str_c(name_prefix, "_doubled_positions.tsv", collapse = "")
  messy_file <- str_c(path[1:n_path - 1], collapse = "/")
  if (length(messy_file) == 0) {
    messy_file <- messy_name
  } else {
    messy_file <- str_c(messy_file, "/", messy_name, collapse = "")
  }
  ambiguous_name <- str_c(name_prefix, "_ambiguous_positions.tsv", collapse = "")
  ambiguous_file <- str_c(path[1:n_path - 1], collapse = "/")
  if (length(ambiguous_file) == 0) {
    ambiguous_file <- ambiguous_name
  } else {
    ambiguous_file <- str_c(ambiguous_file, "/", ambiguous_name, collapse = "")
  }
  if (file.exists(output_file)) file.remove(output_file)
  if (file.exists(messy_file)) file.remove(messy_file)
  if (file.exists(ambiguous_file)) file.remove(ambiguous_file)
  header <- paste('seqnames', 'pos', 'strand', 'reference', 'A', 'C', 'G', 'T', 'coverage', sep = "\t")
  write(header, output_file)
  write(header, ambiguous_file)
  write(header, messy_file)
  
  res <- lapply(chromosome_names, 
                function(chr) fixed_chromosome_positions(bam_file, chr, bam_index, pileup_params, fasta_file, 
                                                         min_coverage, output_file, messy_file, ambiguous_file))
}

# parse command line arguments
parser <- ArgumentParser()
parser$add_argument('-i', '--input', help = 'input BAM file', required = TRUE)
parser$add_argument('-r', '--reference', help = 'reference genome FASTA', required = TRUE)
parser$add_argument('-o', '--output', help = 'output file name', required = TRUE)
parser$add_argument('--minBaseQuality', help = 'minimum QUAL value for each nucleotide in an alignment', type = 'integer')
parser$add_argument('--minMapq', help = 'minimum MAPQ value for an alignment to be included in pileup', type = 'integer')
parser$add_argument('--minNucleotideDepth', help = 'minimum count of each nucleotide at a given position required for said nucleotide to appear in the result', type = 'integer')
parser$add_argument('--minCoverage', help = 'minimum coverage for position to be included in final dataset', type = 'integer')
parser$add_argument('--allChromosomes', help = 'use all chromosomes from BAM (including non-canonical ones)', action = 'store_true')
args <- parser$parse_args()

# get file info
bam_file <- args$input
bam_index <- paste0(bam_file, ".bai")
reference_fasta <- args$reference
output_file <- args$output

# set default filter parameters
max_depth <- 8000 # maximum number of nucleotides to be included in pileup
min_minor_allele_depth <- 0 # get all covered sites
min_base_quality <- 13
min_mapq <- 30
min_nucleotide_depth <- 1
min_coverage <- 0
use_noncanonical <- FALSE

# parse arguments from command line
if (length(args$minBaseQuality) > 0) { min_base_quality <- args$minBaseQuality }
if (length(args$minMapq) > 0) { min_mapq <- args$minMapq }
if (length(args$minNucleotideDepth) > 0) { min_nucleotide_depth <- args$minNucleotideDepth }
if (length(args$minCoverage) > 0) { min_coverage <- args$minCoverage }
use_noncanonical <- args$allChromosomes

# check if bam exists: throw and stop
if (!file.exists(bam_file)) {
  error_message <- paste0(bam_file, " -- BAM file does not exist")
  stop(error_message)
}

# check if index exists: create 
if (!file.exists(bam_index)) {
  indexBam(bam_file)
}

# check if reference genome exists: throw and stop
if (!file.exists(reference_fasta)) {
  error_message <- paste0(reference_fasta, " -- reference FASTA does not exist")
  stop(error_message)
}

# run 
get_positions_from_bam(bam_file, bam_index, reference_fasta, output_file, max_depth, min_base_quality,
                       min_mapq, min_nucleotide_depth, min_minor_allele_depth, min_coverage, use_noncanonical)
