get_params <- function(args) {
  if (length(args) != 4) {
    stop("Error: Incorrect number of arguments. Use -h or --help for usage information.")
  }
  if (!args[1] %in% c("trials", "gp", "remotesensing")) {
    stop("Error: ANALYSIS_TYPE must be either 'trials' or 'gp'. Use -h or --help for usage information.")
  }
  if (!file.exists(args[2])) {
    stop(paste("Error: FNAME_1 does not exist:", args[2]))
  }
  if (!file.exists(args[3])) {
    stop(paste("Error: FNAME_2 does not exist:", args[3]))
  }
  if (!dir.exists(args[4])) {
    stop(paste("Error: DIRNAME_OUTDIR does not exist:", args[4]))
  }
  id_1 <- gsub("-LINEAR.tsv", "", gsub("-MLP.tsv", "", gsub("-TREES.tsv", "", basename(args[2]))))
  id_2 <- gsub("-LINEAR.tsv", "", gsub("-MLP.tsv", "", gsub("-TREES.tsv", "", basename(args[3]))))
  algo_1 <- gsub(".tsv$", "", gsub(paste0("^", id_1, "-"), "", basename(args[2])))
  algo_2 <- gsub(".tsv$", "", gsub(paste0("^", id_2, "-"), "", basename(args[3])))
  if (id_1 != id_2) {
    stop("Error: FNAME_1 and FNAME_2 do not correspond to the same dataset. Use -h or --help for usage information.")
  }
  id <- id_1
  if (args[1] == "trials") {
    fname_trials <- if (grepl("-LINEAR.tsv$", args[2])) {
      args[2]
    } else if (grepl("-LINEAR.tsv$", args[3])) {
      args[3]
    } else {
      NULL
    }
    if (!is.null(fname_trials)) {
      con <- file(fname_trials, "r")
      tmp <- readLines(con, n = 1)
      close(con)
      linear_formula <- gsub("\"", "", unlist(strsplit(tmp[grep("Best model selected: ", tmp)], ": "))[2])
    } else {
      linear_formula <- NA
    }
  } else {
    # "gp" or "remotesensing"
    linear_formula <- NA
  }
  fname_png <- file.path(args[4], paste0(id, "-", algo_1, "_vs_", algo_2, "-COMPARISON.png"))
  fname_tsv <- file.path(args[4], paste0(id, "-", algo_1, "_vs_", algo_2, "-COMPARISON.tsv"))
  list(
    analysis_type = args[1],
    fname_1 = args[2],
    fname_2 = args[3],
    dirname_outdir = args[4],
    id = id,
    algo_1 = algo_1,
    algo_2 = algo_2,
    linear_formula = linear_formula,
    fname_png = fname_png,
    fname_tsv = fname_tsv
  )
}

compare_trial_analyses <- function(params) {
  # args <- c("trials", "tests/tmp/trials/output-australia.soybean-height-MLP.tsv", "tests/tmp/trials/output-australia.soybean-height-LINEAR.tsv", "tests/tmp/trials"); params <- get_params(args)
  # Load the effects from the best linear model
  df_1 <- {
    df <- if ((grepl("-LINEAR.tsv$", params$fname_1))) {
      read.delim(params$fname_1, header = TRUE, skip = 1)
    } else {
      read.delim(params$fname_1, header = TRUE)
    }
    if (length(grep("➵", df$ids)) > 0) {
      df <- df[grep("^gen", df$ids), ]
      df$ids <- gsub("gen➵", "", df$ids)
    }
    df <- data.frame(ids = df$ids, effects = df$effects)
    colnames(df)[2] <- params$algo_1
    df
  }
  df_2 <- {
    df <- if ((grepl("-LINEAR.tsv$", params$fname_2))) {
      read.delim(params$fname_2, header = TRUE, skip = 1)
    } else {
      read.delim(params$fname_2, header = TRUE)
    }
    if (length(grep("➵", df$ids)) > 0) {
      df <- df[grep("^gen", df$ids), ]
      df$ids <- gsub("gen➵", "", df$ids)
    }
    df <- data.frame(ids = df$ids, effects = df$effects)
    colnames(df)[2] <- params$algo_2
    df
  }
  if ((nrow(df_1) == 0) || (nrow(df_2) == 0)) {
    return(1)
  }
  # Merge
  df <- cbind(datasets = gsub("output-", "", params$id), merge(df_1, df_2, by = "ids"))
  # Calculate correlation and R²
  cor_test <- tryCatch(
    cor.test(df[[params$algo_1]], df[[params$algo_2]]),
    error = function(x) {
      list(estimate = NA, p.value = 1.00)
    }
  )
  cor_test = if (is.na(cor_test$estimate)) {
    list(estimate = NA, p.value = 1.00)
  } else {
    cor_test
  }
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
    1 - (sum((df[[params$algo_1]] - df[[params$algo_2]])^2) /

     sum((df[[params$algo_1]] - mean(df[[params$algo_1]]))^2)),
    1 - (sum((df[[params$algo_1]] - df[[params$algo_2]])^2) /
                  sum((df[[params$algo_2]] - mean(df[[params$algo_2]]))^2))
  ))
  # Plot
  png(params$fname_png, type = "cairo")
  par(mar = c(5, 6, 3, 1), mgp = c(4, 1, 0))
  xlab <- if (params$algo_1 == "LINEAR") {
    paste0(
      "Linear Model Estimated Effects\n",
      params$linear_formula
    )
  } else if (params$algo_1 == "TREES") {
    "XGBoost Marginal Effects (SHAP method)"
  } else {
    "MLP Marginal Effects (Perturbation method)"
  }
  ylab <- if (params$algo_2 == "LINEAR") {
    paste0(
      "Linear Model Estimated Effects\n",
      params$linear_formula
    )
  } else if (params$algo_2 == "TREES") {
    "XGBoost Marginal Effects (SHAP method)"
  } else {
    "MLP Marginal Effects (Perturbation method)"
  }
  plot(df[[params$algo_1]], df[[params$algo_2]],
    xlab = xlab,
    ylab = ylab,
    main = params$id
  )
  grid()
  text(min(df[[params$algo_1]]), max(df[[params$algo_2]]),
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

compare_gp_or_remotesensing_analyses <- function(params) {
  # args <- c("gp", "tests/tmp/gp/output-sorghum-YLD-LINEAR.tsv", "tests/tmp/gp/output-sorghum-YLD-MLP.tsv", "tests/tmp/gp"); params <- get_params(args)
  # args <- c("gp", "tests/tmp/gp/output-test-YLD-TREES.tsv", "tests/tmp/gp/output-test-YLD-MLP.tsv", "tests/tmp/gp"); params <- get_params(args)
  df_1 <- {
    df <- read.delim(params$fname_1, sep = "\t", header = TRUE)
    data.frame(
      datasets = gsub(".tsv", "", df$datasets),
      reps = df$reps,
      folds = df$folds,
      nt = df$nt,
      nv = df$nv,
      models = df$models,
      corr = df$corr,
      r2 = df$r2
    )
  }
  df_2 <- {
    df <- read.delim(params$fname_2, sep = "\t", header = TRUE)
    data.frame(
      datasets = gsub(".tsv", "", df$datasets),
      reps = df$reps,
      folds = df$folds,
      nt = df$nt,
      nv = df$nv,
      models = df$models,
      corr = df$corr,
      r2 = df$r2
    )
  }
  df <- rbind(df_1, df_2)
  if ((gsub("output-", "", params$id) != gsub(".tsv", "", unique(df_1$datasets))) || (gsub("output-", "", params$id) != gsub(".tsv", "", unique(df_2$datasets)))) {
    stop("Error: Dataset ID in linear and/or MLP results does not match the expected dataset ID.")
  }
  png(params$fname_png, width = length(unique(df$models)) * 300)
  par(mfrow=c(2, 1))
  boxplot(corr ~ models, data = df, xlab = "", ylab = "Pearson's Correlation")
  grid()
  boxplot(r2 ~ models, data = df, xlab = "", ylab = "Coefficient of Determination (R²)")
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
args <- commandArgs(trailingOnly = TRUE)
if (args[1] == "-h" || args[1] == "--help") {
  cat("Usage: Rscript comparison.R ANALYSIS_TYPE FNAME_1 FNAME_2 DIRNAME_OUTDIR\n")
  cat("Arguments:\n")
  cat("\t1. ANALYSIS_TYPE:\n")
  cat("\t\t+ 'trials' for extracting marginal effects of each genotype, or\n")
  cat("\t\t+ 'gp' for repeated k-fold cross-validation for genomic prediction.\n")
  cat("\t\t+ 'remotesensing' for repeated k-fold cross-validation for remote sensing.\n")
  cat("\t2. FNAME_1: The file name for one model's results (e.g. linear, xgboost or mlp).\n")
  cat("\t3. FNAME_2: The file name for another model's results (e.g. linear, xgboost or mlp).\n")
  cat("\t4. DIRNAME_OUTDIR: The output directory name.\n")
  quit(status = 0)
}
# Testing: source("tests/scripts/comparison.R")
# args <- c("trials", "~/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-LINEAR.tsv", "~/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-MLP.tsv", "~/Documents/mlp/tests/tmp/trials")
# args <- c("gp", "~/Documents/mlp/tests/tmp/gp/output-sorghum-HT-LINEAR.tsv", "~/Documents/mlp/tests/tmp/gp/output-sorghum-HT-MLP.tsv", "~/Documents/mlp/tests/tmp/gp")
# args <- c("trials", "~/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-LINEAR.tsv", "~/Documents/mlp/tests/tmp/trials/output-ilri.sheep-birthwt-TREES.tsv", "~/Documents/mlp/tests/tmp/trials")
# args <- c("gp", "~/Documents/mlp/tests/tmp/gp/output-sorghum-HT-LINEAR.tsv", "~/Documents/mlp/tests/tmp/gp/output-sorghum-HT-TREES.tsv", "~/Documents/mlp/tests/tmp/gp")
# args <- c("trials", "~/Documents/mlp/tests/tmp/trials/output-australia.soybean-yield-LINEAR.tsv", "~/Documents/mlp/tests/tmp/trials/output-australia.soybean-yield-TREES.tsv", "~/Documents/mlp/tests/tmp/trials")
# args <- c("trials", "~/Documents/mlp/tests/tmp/trials/output-australia.soybean-yield-MLP.tsv", "~/Documents/mlp/tests/tmp/trials/output-australia.soybean-yield-TREES.tsv", "~/Documents/mlp/tests/tmp/trials")
# args <- c("gp", "~/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1-LINEAR.tsv", "~/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1-TREES.tsv", "~/Documents/mlp/tests/tmp/gp")
# args <- c("gp", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_2-LINEAR.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_2-MLP.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp")
# args <- c("gp", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_2-LINEAR.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_2-TREES.tsv", "/home/jp3h/Documents/mlp/tests/tmp/gp")
params <- get_params(args)
if (params$analysis_type == "trials") {
  compare_trial_analyses(params)
} else {
  # "gp" or "remotesensing"
  compare_gp_or_remotesensing_analyses(params)
}