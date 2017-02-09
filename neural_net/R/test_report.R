library(readr)
library(ggplot2)
library(dplyr)
library(tidyr)

data_file_name <- "/Users/bioinformaticshub/Documents/Ira/soft/neural_net/50nodes_test.tsv"
result_data <- read_tsv(data_file_name)

# SE
result_data %>%
  select(ends_with("_se")) %>%
  gather(ends_with("_se"), key = "nucleotide", value = "squared_error") %>%
  ggplot() +
  geom_boxplot(aes(nucleotide, squared_error, colour = nucleotide)) +
  theme_bw() +
  labs(title = 'Squared error (50 nodes NN)')

result_data %>%
  select(ends_with("_se")) %>%
  gather(ends_with("_se"), key = "nucleotide", value = "squared_error") %>%
  filter(squared_error < 1000) %>%
  filter(squared_error > 0) %>%
  ggplot() +
  geom_boxplot(aes(nucleotide, squared_error, colour = nucleotide)) +
  theme_bw() +
  labs(title = 'Squared error (50 nodes NN)')

summary(result_data$A_se)
summary(result_data$G_se)
summary(result_data$C_se)
summary(result_data$T_se)

result_data %>%
  gather(ends_with("_se"), key = "nucleotide", value = "squared_error") %>%
  select(squared_error) %>%
  summary()

# residual noise
result_data %>%
  ggplot() +
  geom_boxplot(aes(1, residual_noise)) +
  theme_bw() +
  labs(title = 'Residual noise (50 nodes NN)')
summary(result_data$residual_noise)
# % of positions with non-zero residual noise
100 * nrow(filter(result_data, residual_noise > 0)) / nrow(result_data)

# reference position error
result_data %>%
  ggplot() +
  geom_boxplot(aes(1, reference_error)) +
  theme_bw() +
  labs(title = 'Reference position error (50 nodes NN)')
result_data %>%
  filter(abs(reference_error) < 100) %>%
  ggplot() +
  geom_boxplot(aes(1, reference_error)) +
  theme_bw() +
  labs(title = 'Reference position error (50 nodes NN)')
summary(result_data$reference_error)
