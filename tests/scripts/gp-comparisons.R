fnames_tmp <- list.files(".", pattern = ".*.tsv")
file_names_input <- fnames_tmp[!grepl("^output", fnames_tmp)]
file_names_linear <- fnames_tmp[grepl("^output.*-LINEAR.*.tsv", fnames_tmp)]
file_names_mlp <- fnames_tmp[grepl("^output.*-MLP.*.tsv", fnames_tmp)]

df_results = NULL
for (file_name_input in file_names_input) {
  # file_name_input <- file_names_input[1]
  id <- gsub("input_simulated-", "", gsub(".tsv", "", file_name_input))
  idx_linear <- grep(id, file_names_linear)
  idx_mlp <- grep(id, file_names_mlp)
  if ((length(idx_linear) == 0) || (length(idx_mlp) == 0)) {
    next
  }
  file_name_linear <- file_names_linear[tail(idx_linear, 1)]
  file_name_mlp <- file_names_mlp[tail(idx_mlp, 1)]

  df_linear <- read.delim(file_name_linear, sep = "\t", header = TRUE)
  df_mlp <- read.delim(file_name_mlp, sep = "\t", header = TRUE)
  # df_linear_tmp = aggregate(corr ~ reps + models, FUN=mean, data=df_linear)
  # df_mlp_tmp = aggregate(corr ~ reps + models, FUN=mean, data=df_mlp)
  df_merged <- rbind(
    merge(aggregate(corr ~ models, FUN = mean, data = df_linear), aggregate(corr ~ models, FUN = sd, data = df_linear), by = "models"),
    merge(aggregate(corr ~ models, FUN = mean, data = df_mlp), aggregate(corr ~ models, FUN = sd, data = df_mlp), by = "models")
  )
  colnames(df_merged) <- c("models", "mean_corr", "sd_corr")

  df_results <- if (is.null(df_results)) {
    cbind(id, df_merged)
  } else {
    rbind(df_results, cbind(id, df_merged))
  }
}

print(df_results)

n <- length(unique(df_results$models))
m <- length(unique(df_results$id))
M <- matrix(df_results$mean_corr, nrow = n, ncol = m, byrow = FALSE)
rownames(M) <- unique(df_results$models)
colnames(M) <- unique(df_results$id)
colours_models <- c(
  # "#8dd3c7",
  # "#ffffb3",
  # "#bebada",
  # "#80b1d3",
  # "#fb8072"
  "#0072b2",
  "#f0e442",
  "#cc79a7",
  "#009e73",
  "#d55e00"
)

png("gp-comparisons-simulated.png", height = 700, width = 900)
barplot(
  M,
  beside = TRUE,
  col = colours_models,
  border = NA,
  legend.text = TRUE,
  args.legend = list(x = "topleft"),
  xlab = "Dataset",
  ylab = "Pearson's correlation"
)
grid()
dev.off()
