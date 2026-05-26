fnames_tmp <- list.files(".", pattern = ".*.tsv")
file_names_input <- fnames_tmp[!grepl("^output", fnames_tmp)]
file_names_linear <- fnames_tmp[grepl("^output.*-LINEAR.*.tsv", fnames_tmp)]
file_names_mlp <- fnames_tmp[grepl("^output.*-MLP.*.tsv", fnames_tmp)]

for (file_name_input in file_names_input) {
  # file_name_input <- file_names_input[1]
  id <- gsub("input_simulated-", "", gsub(".tsv", "", file_name_input))
  idx_linear <- grep(id, file_names_linear)
  idx_mlp <- grep(id, file_names_mlp)
  if ((length(idx_linear) == 0) || (length(idx_mlp) == 0)) {
    next
  }
  fname_linear <- file_names_linear[tail(idx_linear, 1)]
  fname_mlp <- file_names_mlp[tail(idx_mlp, 1)]
  # Load the effects from the best linear model
  df_linear <- read.delim(fname_linear, TRUE)
  if (length(grep("➵", df_linear$ids)) > 0) {
    df_linear <- df_linear[grep("^gen", df_linear$ids), ]
    df_linear$ids <- gsub("gen➵", "", df_linear$ids)
  }
  colnames(df_linear)[2] <- "linear"
  # Load the marginal effects from mlp
  df_mlp <- read.delim(fname_mlp, TRUE)
  df_mlp <- df_mlp[grep("^gen", df_mlp$ids), 1:2]
  df_mlp$ids <- gsub("gen➵", "", df_mlp$ids)
  colnames(df_mlp)[2] <- "mlp"
  if ((nrow(df_linear) == 0) || (nrow(df_mlp) == 0)) {
    next
  }
  # Merge
  df <- merge(df_linear, df_mlp, by = "ids")
  # Calculate correlation and R²
  cor_test <- tryCatch(
    cor.test(df$linear, df$mlp),
    error = function(x) {
      list(estimate = NA, p.value = 1.00)
    }
  )
  print(cor_test)
  annot <- if (cor_test$p.value < 0.0001) {
    "***"
  } else if (cor_test$p.value < 0.001) {
    "**"
  } else if (cor_test$p.value < 0.01) {
    "*"
  } else {
    "ns"
  }
  r_squared <- mean(c(
    1 - (sum((df$linear - df$mlp)^2) /
                  sum((df$linear - mean(df$linear))^2)),
    1 - (sum((df$linear - df$mlp)^2) /
                  sum((df$mlp - mean(df$mlp))^2))
  ))
  # Plot
  file_name_png <- paste0("comparison-", id, ".png")
  linear_model_formula <- gsub(
    paste0("output-", id, "-LINEAR_"), "",
    gsub("output_simulated-", "", gsub(id, "", gsub("-LINEAR_", "", gsub(",random=~", ",\nrandom=~", gsub(",data=df", "", gsub(",trace=FALSE", "", gsub(".tsv", "", fname_linear)))))))
  )
  png(file_name_png, type="cairo")
  par(mar=c(5, 6, 3, 1), mgp=c(4, 1, 0))
  plot(df$linear, df$mlp,
    xlab = paste0(
      "Linear Model Estimated Effects\n",
      linear_model_formula
    ),
    ylab = "Multi-layer Perceptron\nMarginal Effects",
    main = id
  )
  grid()
  text(min(df$linear), max(df$mlp),
    label = paste0(
      "\n\n\ncor=", round(100 * cor_test$estimate, 2), "%",
      annot, "\nR²=", round(r_squared, 2),
      "\nn=", nrow(df)
    ),
    pos = c(4, 1)
  )
  dev.off()
}
