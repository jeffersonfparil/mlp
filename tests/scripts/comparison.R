args <- commandArgs(trailingOnly = TRUE)
if (args[1] == "-h" || args[1] == "--help") {
  cat("Usage: Rscript comparison.R ANALYSIS_TYPE FNAME_LINEAR FNAME_MLP DIRNAME_OUTDIR\n")
  cat("Arguments:\n")
  cat("\t1. ANALYSIS_TYPE:\n")
  cat("\t\t+ 'trials' for extracting marginal effects of each genotype, or\n")
  cat("\t\t+ 'gp' for repeated k-fold cross-validation for genomic prediction.\n")
  cat("\t2. FNAME_LINEAR: The file name for the linear model results.\n")
  cat("\t3. FNAME_MLP: The file name for the MLP model results.\n")
  cat("\t4. DIRNAME_OUTDIR: The output directory name.\n")
  quit(status = 0)
}

get_params <- function(args) {
  if (length(args) != 4) {
    stop("Error: Incorrect number of arguments. Use -h or --help for usage information.")
  }
  if (!args[1] %in% c("trials", "gp")) {
    stop("Error: ANALYSIS_TYPE must be either 'trials' or 'gp'. Use -h or --help for usage information.")
  }
  if (!file.exists(args[2])) {
    stop(paste("Error: FNAME_LINEAR does not exist:", args[2]))
  }
  if (!file.exists(args[3])) {
    stop(paste("Error: FNAME_MLP does not exist:", args[3]))
  }
  if (!dir.exists(args[4])) {
    stop(paste("Error: DIRNAME_OUTDIR does not exist:", args[4]))
  }
  id_linear <- gsub("-LINEAR.tsv", "", basename(args[2]))
  id_mlp <- gsub("-MLP.tsv", "", basename(args[3]))
  if (id_linear != id_mlp) {
    stop("Error: FNAME_LINEAR and FNAME_MLP do not correspond to the same dataset. Use -h or --help for usage information.")
  }
  id <- id_linear
  if (args[1] == "trials") {
    fname_log_linear <- file.path(dirname(args[2]), gsub("^output-", "linear_analysis-", gsub("-LINEAR.tsv", ".log", basename(args[2]))))
    if (!file.exists(fname_log_linear)) {
      stop(paste("Error: Log file for linear analysis does not exist:", fname_log_linear))
    }
    tmp <- readLines(fname_log_linear)
    linear_formula <- gsub("\"", "", unlist(strsplit(tmp[grep("Best model selected: ", tmp)], ": "))[2])
  } else if (args[1] == "gp") {
    linear_formula = NA
  }
  fname_png <- file.path(args[4], paste0(id, "-COMPARISON.png"))
  fname_tsv <- file.path(args[4], paste0(id, "-COMPARISON.tsv"))
  list(
    analysis_type = args[1],
    fname_linear = args[2],
    fname_mlp = args[3],
    dirname_outdir = args[4],
    id = id,
    linear_formula = linear_formula,
    fname_png = fname_png,
    fname_tsv = fname_tsv
  )
}

compare_trial_analyses <- function(params) {
  # args <- c("trials", "tests/tmp/trials/output-australia.soybean-height-LINEAR.tsv", "tests/tmp/trials/output-australia.soybean-height-MLP.tsv", "tests/tmp/trials"); params <- get_params(args)
  # Load the effects from the best linear model
  df_linear <- read.delim(params$fname_linear, TRUE)
  if (length(grep("➵", df_linear$ids)) > 0) {
    df_linear <- df_linear[grep("^gen", df_linear$ids), ]
    df_linear$ids <- gsub("gen➵", "", df_linear$ids)
  }
  colnames(df_linear)[2] <- "linear"
  # Load the marginal effects from mlp
  df_mlp <- read.delim(params$fname_mlp, TRUE)
  df_mlp <- df_mlp[grep("^gen", df_mlp$ids), 1:2]
  df_mlp$ids <- gsub("gen➵", "", df_mlp$ids)
  colnames(df_mlp)[2] <- "mlp"
  if ((nrow(df_linear) == 0) || (nrow(df_mlp) == 0)) {
    next
  }
  # Merge
  df <- cbind(datasets = gsub("output-", "", params$id), merge(df_linear, df_mlp, by = "ids"))
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
  png(params$fname_png, type="cairo")
  par(mar=c(5, 6, 3, 1), mgp=c(4, 1, 0))
  plot(df$linear, df$mlp,
    xlab = paste0(
      "Linear Model Estimated Effects\n",
      params$linear_formula
    ),
    ylab = "Multi-layer Perceptron\nMarginal Effects",
    main = params$id
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
  write.table(df, params$fname_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("Output:\n")
  cat("\t- Scatterplot: ", params$fname_png, "\n")
  cat("\t- Data: ", params$fname_tsv, "\n")
}

compare_gp_analyses <- function(params) {
  # args <- c("gp", "tests/tmp/gp/output-sorghum-YLD-LINEAR.tsv", "tests/tmp/gp/output-sorghum-YLD-MLP.tsv", "tests/tmp/gp"); params <- get_params(args)
  df_linear <- read.delim(params$fname_linear, sep = "\t", header = TRUE)
  df_mlp <- {
    df_mlp <- read.delim(params$fname_mlp, sep = "\t", header = TRUE)
    activation <- paste(sort(unique(df_mlp$activation)), collapse = "_")
    weights_initialisation <- paste(sort(unique(df_mlp$weights_initialisation)), collapse = "_")
    optimiser <- paste(sort(unique(df_mlp$optimiser)), collapse = "_")
    n_hidden_layers <- mean(df_mlp$n_hidden_layers)
    n_hidden_nodes <- mean(df_mlp$n_hidden_nodes)
    n_epochs <- mean(df_mlp$n_epochs)
    n_validation <- mean(df_mlp$n_validation)
    df_mlp[, colnames(df_mlp) == "models"] <- paste(c("mlp", activation, weights_initialisation, optimiser, paste(c("H", "N", "E", "V"), c(n_hidden_layers, n_hidden_nodes, n_epochs, n_validation), sep = "_")), collapse = "-")
    idx_cols_mlp <- c()
    for (col in colnames(df_linear)) {
      idx <- grep(col, colnames(df_mlp))
      if (length(idx) == 0) {
        stop(paste("Error: Column", col, "in linear results not found in MLP results."))
      }
      idx_cols_mlp <- c(idx_cols_mlp, idx[1])
    }
    df_mlp[, idx_cols_mlp]
  }
  df <- rbind(df_linear, df_mlp)
  if ((gsub("output-", "", params$id) != gsub(".tsv", "", unique(df_linear$datasets))) || (gsub("output-", "", params$id) != gsub(".tsv", "", unique(df_mlp$datasets)))) {
    stop("Error: Dataset ID in linear and/or MLP results does not match the expected dataset ID.")
  }
  df$datasets <- gsub("output-", "", params$id)
  df$models <- as.factor(df$models)
  # aggregate(corr ~ models, FUN = mean, data = df)
  # aggregate(corr ~ models, FUN = sd, data = df)
  png(params$fname_png, width = nlevels(df$models) * 300)
  boxplot(corr ~ models, data = df, xlab = "", ylab = "Pearson's Correlation")
  grid()
  dev.off()
  write.table(df, params$fname_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
  cat("Output:\n")
  cat("\t- Boxplot: ", params$fname_png, "\n")
  cat("\t- Data: ", params$fname_tsv, "\n")
}

###########################################################
# Execute
###########################################################
# Testing: source("tests/scripts/comparison.R")
# args <- c("trials", "/home/jp3h/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-LINEAR.tsv", "/home/jp3h/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-MLP.tsv", "/home/jp3h/Documents/mlp/tests/tmp/trials")
# args <- c("gp", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-sorghum-HT-LINEAR.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-sorghum-HT-MLP.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp")
params <- get_params(args)
if (params$analysis_type == "trials") {
  compare_trial_analyses(params)
} else {
  compare_gp_analyses(params)
}