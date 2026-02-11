#!/usr/bin/env Rscript
# analyze.R — Query PostgreSQL and produce stats + plots for sound textures

library(DBI)
library(RPostgres)
library(ggplot2)

# --- Database connection ---
con <- dbConnect(
  Postgres(),
  dbname   = "sound_textures",
  host     = "localhost",
  port     = 5432,
  user     = "postgres",
  password = Sys.getenv("DB_PASSWORD", "")
)

textures <- dbReadTable(con, "textures")
dbDisconnect(con)

cat("Loaded", nrow(textures), "textures from database.\n\n")

# --- Statistical summaries ---
numeric_cols <- c("base_frequency", "duration", "amplitude",
                  "filter_cutoff", "mod_rate", "mod_depth")

cat("=== Statistical Summary ===\n")
for (col in numeric_cols) {
  vals <- textures[[col]]
  cat(sprintf("\n%s:\n", col))
  cat(sprintf("  mean   = %.3f\n", mean(vals)))
  cat(sprintf("  median = %.3f\n", median(vals)))
  cat(sprintf("  sd     = %.3f\n", sd(vals)))
  cat(sprintf("  min    = %.3f\n", min(vals)))
  cat(sprintf("  max    = %.3f\n", max(vals)))
}

cat("\n=== Correlation Matrix ===\n")
cor_matrix <- cor(textures[, numeric_cols])
print(round(cor_matrix, 3))

# --- Counts by noise type ---
cat("\n=== Counts by Noise Type ===\n")
print(table(textures$noise_type))

# --- Create plots directory ---
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)
plots_dir <- file.path(script_dir, "plots")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

# --- Plot 1: Histogram of base frequencies ---
p1 <- ggplot(textures, aes(x = base_frequency)) +
  geom_histogram(bins = 15, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Base Frequencies",
       x = "Base Frequency (Hz)", y = "Count") +
  theme_minimal()
ggsave(file.path(plots_dir, "hist_base_frequency.png"), p1,
       width = 7, height = 5)

# --- Plot 2: Histogram of durations ---
p2 <- ggplot(textures, aes(x = duration)) +
  geom_histogram(bins = 12, fill = "coral", color = "white") +
  labs(title = "Distribution of Durations",
       x = "Duration (s)", y = "Count") +
  theme_minimal()
ggsave(file.path(plots_dir, "hist_duration.png"), p2,
       width = 7, height = 5)

# --- Plot 3: Scatter — filter cutoff vs modulation rate ---
p3 <- ggplot(textures, aes(x = filter_cutoff, y = mod_rate,
                            color = noise_type)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Filter Cutoff vs Modulation Rate",
       x = "Filter Cutoff (Hz)", y = "Modulation Rate (Hz)",
       color = "Noise Type") +
  theme_minimal()
ggsave(file.path(plots_dir, "scatter_cutoff_vs_modrate.png"), p3,
       width = 7, height = 5)

# --- Plot 4: Box plots of amplitude by noise type ---
p4 <- ggplot(textures, aes(x = noise_type, y = amplitude,
                            fill = noise_type)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Amplitude by Noise Type",
       x = "Noise Type", y = "Amplitude") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(plots_dir, "box_amplitude_by_noise.png"), p4,
       width = 7, height = 5)

# --- Plot 5: Box plots of filter cutoff by noise type ---
p5 <- ggplot(textures, aes(x = noise_type, y = filter_cutoff,
                            fill = noise_type)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Filter Cutoff by Noise Type",
       x = "Noise Type", y = "Filter Cutoff (Hz)") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(plots_dir, "box_cutoff_by_noise.png"), p5,
       width = 7, height = 5)

cat("\nPlots saved to", plots_dir, "\n")
