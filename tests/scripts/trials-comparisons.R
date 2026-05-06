file_names_input <- list.files(".", pattern = "input")
file_names_linear <- list.files(".", pattern = "output_.*-LINEAR")
file_names_mlp <- list.files(".", pattern = "output_.*-MLP")

for (file_name_input in file_names_input) {
  # file_name_input <- file_names_input[1]
  id <- gsub("input_simulated-", "", gsub(".tsv", "", file_name_input))
  file_name_linear <- file_names_linear[grep(id, file_names_linear)]
  file_name_mlp <- file_names_mlp[grep(id, file_names_mlp)]

  # Load the effects from the best linear model
  df_linear <- read.delim(file_name_linear, TRUE)
  if (length(grep("➵", df_linear$ids)) > 0) {
    df_linear <- df_linear[grep("^gen", df_linear$ids), ]
    df_linear$ids <- gsub("gen➵", "", df_linear$ids)
  }
  colnames(df_linear)[2] <- "linear"

  # Load the marginal effects from mlp
  df_mlp <- read.delim(file_name_mlp, TRUE)
  df_mlp <- df_mlp[grep("^gen", df_mlp$ids), 1:2]
  df_mlp$ids <- gsub("gen➵", "", df_mlp$ids)
  colnames(df_mlp)[2] <- "mlp"

  # Merge
  df <- merge(df_linear, df_mlp, by = "ids")

  # Calculate correlation and R²
  cor_test <- cor.test(df$linear, df$mlp)
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
    paste0("output_simulated-", id, "-LINEAR_"), "",
      gsub(".tsv", "", file_name_linear)
  )
  png(file_name_png)
  plot(df$linear, df$mlp,
    xlab = paste0(
        "Linear Model Estimated Effects\n(",
        linear_model_formula, ")"
    ),
    ylab = "Multi-layer Perceptron Marginal Effects",
    main = id
  )
  grid()
  text(min(df$linear), max(df$mlp),
    label = paste0(
        "\n\ncor=", round(100 * cor_test$estimate, 2), "%",
        annot, "\nR²=", round(r_squared, 2)
    ),
      pos = c(4, 1)
  )
  dev.off()
}
