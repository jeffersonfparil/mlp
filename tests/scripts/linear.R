args <- commandArgs(trailingOnly = TRUE)
if (args[1] == "-h" || args[1] == "--help") {
  cat("Usage: Rscript empiricalprep.R ANALYSIS_TYPE FNAME_INPUT DIRNAME_OUTPUT\n")
  cat("Arguments:\n")
  cat("\t1. ANALYSIS_TYPE:\n")
  cat("\t\t+ 'trials' for extracting marginal effects of each genotype, or\n")
  cat("\t\t+ 'gp' for repeated k-fold cross-validation for genomic prediction.\n")
  cat("\t2. FNAME_INPUT: The input file name.\n")
  cat("\t\t+ For 'trials' analysis, this should be a tab-separated file with a header row and columns for year, site, treatment, entry, replication, and response variable.\n")
  cat("\t\t+ For 'gp' analysis, this should be a tab-separated file with a header row and columns for the response variable followed by the features.\n")
  cat("\t3. DIRNAME_OUTPUT: The output directory.\n")
  cat("For trials analysis, additional arguments are:\n")
  cat("\t\t4. is_simulated: TRUE/FALSE (Default: FALSE)\n")
  cat("\t\t5. exclude_lm: TRUE/FALSE (Default: FALSE)\n")
  cat("\t\t6. exclude_sommer: TRUE/FALSE (Default: FALSE)\n")
  cat("\t\t7. verbose: TRUE/FALSE (Default: FALSE)\n")
  cat("For genomic prediction analysis, additional arguments are:\n")
  cat("\t\t4. n_reps: numeric (Default: 1)\n")
  cat("\t\t5. n_folds: numeric (Default: 5)\n")
  cat("\t\t6. n_iterations: numeric (Default: 1000)\n")
  cat("\t\t7. n_burnin_iterations: numeric (Default: 100)\n")
  cat("\t\t8. models: comma-separated list of 'BRR', 'BayesA', 'BayesB', 'BayesC' (Default: 'BRR,BayesA,BayesB,BayesC')\n")
  cat("\t\t9. base_seed: numeric (Default: 123)\n")
  cat("\t\t10. verbose: TRUE/FALSE (Default: FALSE)\n")
  cat("Examples:\n")
  cat("DIR=${HOME}/Documents/mlp/tests\n")
  cat("cd ${DIR}/scripts\n")
  cat("mkdir tmp\n")
  cat("###############\n")
  cat("TRIALS ANALYSIS\n")
  cat("###############\n")
  cat("Rscript empiricalprep.R trials ${DIR}/datasets/agridat/australia.soybean.txt tmp\n")
  cat("Rscript linear.R trials tmp/australia.soybean-yield.tsv tmp FALSE FALSE TRUE TRUE\n")
  cat("###########################\n")
  cat("GENOMIC PREDICTION ANALYSIS\n")
  cat("###########################\n")
  cat("MLP=${HOME}/Documents/mlp/target/release/mlp\n")
  cat("sh simulate.sh $MLP gp tmp BINARY 100 50 2\n")
  cat("Rscript linear.R gp tmp/simulated-DATA_TYPE_BINARY-N_100-P_50-HIDDEN_LAYERS_2.tsv tmp 5 10 100 10 'BRR,BayesA' 123 TRUE\n")
  quit(status = 0)
}

library("stringr")
library("lme4")
if (nzchar(system.file(package = "asreml"))) {
  library("asreml")
}
if (nzchar(system.file(package = "sommer"))) {
  library("sommer") # Currently, too slow for non-trivialy large datasets, e.g. 21,000 observations simulated here
}
library(BGLR)

#' Parse command-line arguments for the linear model fitting and effect extraction process, including analysis type, input file, and model fitting parameters.
get_params <- function(args) {
  # args <- c("trials", "input-file.tsv", "tmp", "TRUE", "TRUE", "TRUE", "TRUE")
  # args <- c("gp", "input-file.tsv", "tmp", "6", "7", "123", "4567")
  analysis_type <- args[1]
  if (!(analysis_type %in% c("trials", "gp"))) {
    stop("ERROR: Invalid analysis type. Must be either 'trials' or 'gp'.")
  }
  if (analysis_type == "trials" && length(args) < 7) {
    stop(paste0("ERROR: For 'trials' analysis, 7 arguments are required:\n",
      "\t1. analysis_type",
      "\t2. input file name",
      "\t3. output directory",
      "\t4. is_simulated (TRUE/FALSE)",
      "\t5. exclude_lm (TRUE/FALSE)",
      "\t6. exclude_sommer (TRUE/FALSE)",
      "\t7. verbose (TRUE/FALSE)"
    ))
  }
  if (analysis_type == "gp" && length(args) < 10) {
    stop(paste0("ERROR: For 'gp' analysis, 10 arguments are required:\n",
      "\t1. analysis_type",
      "\t2. input file name",
      "\t3. output directory",
      "\t4. n_reps (numeric)",
      "\t5. n_folds (numeric)",
      "\t6. n_iterations (numeric)",
      "\t7. n_burnin_iterations (numeric)",
      "\t8. models (comma-separated list of 'BRR', 'BayesA', 'BayesB', 'BayesC')",
      "\t9. base_seed (numeric)",
      "\t10. verbose (TRUE/FALSE)"
    ))
  }

  fname_input <- args[2]
  if (!file.exists(fname_input)) {
    stop(paste0("ERROR: The input file '", fname_input, "' does not exist."))
  } else {
    extension_names <- rev(unlist(strsplit(args[2], split = "\\.")))[1]
    if (extension_names != "tsv") {
      stop(paste0("ERROR: The input file '", fname_input, "' must be a TSV file with a .tsv extension."))
    }
  }
  dirname_output <- args[3]
  if (!dir.exists(dirname_output)) {
    stop(paste0("ERROR: The output directory '", dirname_output, "' does not exist."))
  }
  params <- list(
    analysis_type = analysis_type,
    fname_input = fname_input,
    dirname_output = dirname_output,
    trials = list(
      is_simulated = NA,
      exclude_lm = NA,
      exclude_sommer = NA,
      verbose = NA
    ),
    gp = list(
      n_folds = NA,
      n_reps = NA,
      n_iterations  = NA,
      n_burnin_iterations = NA,
      models = NA,
      base_seed = NA,
      verbose = NA
    )
  )
  suppressWarnings(
    if (analysis_type == "trials") {
      params$trials$is_simulated <- if (args[4] == "TRUE") TRUE else FALSE
      params$trials$exclude_lm <- if (args[5] == "TRUE") TRUE else FALSE
      params$trials$exclude_sommer <- if (args[6] == "TRUE") TRUE else FALSE
      params$trials$verbose <- if (args[7] == "TRUE") TRUE else FALSE
    } else if (analysis_type == "gp") {
      params$gp$n_folds <- if (!is.na(args[5]) && !is.na(as.numeric(args[5]))) as.numeric(args[5]) else 5
      params$gp$n_reps <- if (!is.na(args[4]) && !is.na(as.numeric(args[4]))) as.numeric(args[4]) else 5
      params$gp$n_iterations <- if (!is.na(args[6]) && !is.na(as.numeric(args[6]))) as.numeric(args[6]) else 6000
      params$gp$n_burnin_iterations <- if (!is.na(args[7]) && !is.na(as.numeric(args[7]))) as.numeric(args[7]) else 1000
      params$gp$models <- {
        models <- if (!is.na(args[8])) strsplit(args[8], split = ",")[[1]] else c("BRR", "BayesA", "BayesB", "BayesC")
        for (m in models) {
          if (!(m %in% c("BRR", "BayesA", "BayesB", "BayesC"))) {
            stop(paste0("ERROR: Invalid model '", m, "'. Valid models are 'BRR', 'BayesA', 'BayesB', and 'BayesC'."))
          }
        }
        models
      }
      params$gp$base_seed <- if (!is.na(args[9]) && !is.na(as.numeric(args[9]))) as.numeric(args[9]) else ceiling(100*runif(1))
      params$gp$verbose <- if (args[10] == "TRUE") TRUE else FALSE
    }
  )
  params
}

#' Process features in the dataframe by converting explanatory variables to factors and creating a dummy environment variable if needed.
#' Importantly, we assume that the first column of the dataframe is the numeric response variable (y) and the rest are non-numeric explanatory variables. The function also identifies the feature names for modeling.
process_features <- function(df) {
  # fname = list.files(path=".", pattern=".tsv$")[19]; df = read.table(fname, sep="\t", header=TRUE, na.strings=c("", "NA", "NAN", "NaN", "na", "nan"))
  # Assuming only the first column is the numeric response variable and the rest are non-numeric explanatory variables
  ids_features <- colnames(df)[2:ncol(df)]
  for (j in 2:ncol(df)) {
    df[, j] <- as.factor(df[, j])
  }
  if (length(ids_features) > 2) {
    idx <- which((names(df) != "y") & (names(df) != "gen"))
    df$dummy_env <- apply(df[, idx, drop = FALSE], MARGIN = 1, FUN = function(x) {paste(x, collapse="|")})
    ids_features <- c(ids_features, "dummy_env")
  }
  list(df = df, ids_features=ids_features)
}

#' Calculate AIC for lm, lmerMod (lme4), asreml, or mmes (sommer) models.
aic_lm_lmer_asreml <- function(mod) {
  # mod = model_candidates[[32]]
  if ((class(mod) == "lm") || (class(mod) == "lmerMod")) {
    AIC(mod)
  } else if (class(mod) == "asreml") {
    -2 * mod$loglik + 2 * nrow(summary(mod)$varcomp)
  } else if (class(mod) == "mmes") {
    -mod$AIC # not sure why sommer generates negative AIC & BIC and positive loglik
  } else {
    # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
    NA
  }
}

#' Calculate BIC for lm, lmerMod (lme4), asreml, or mmes (sommer) models.
bic_lm_lmer_asreml <- function(mod) {
  # mod = model_candidates[[13]]
  if ((class(mod) == "lm") || (class(mod) == "lmerMod")) {
    BIC(mod)
  } else if (class(mod) == "asreml") {
    -2 * mod$loglik + nrow(summary(mod)$varcomp) * log(summary(mod)$nedf)
  } else if (class(mod) == "mmes") {
    -mod$BIC # not sure why sommer generates negative AIC & BIC and positive loglik
  } else {
    # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
    NA
  }
}

#' Calculate log-likelihood for lm, lmerMod (lme4), asreml, or mmes (sommer) models.
loglik_lm_lmer_asreml <- function(mod) {
  # mod = model_candidates[[1]]
  if ((class(mod) == "lm") || (class(mod) == "lmerMod")) {
    as.numeric(logLik(mod))
  } else if (class(mod) == "asreml") {
    mod$loglik
  } else if (class(mod) == "mmes") {
    -tail(mod$llik[1, ], 1) # not sure why sommer generates negative AIC & BIC and positive loglik
  } else {
    # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
    NA
  }
}

#' Generate model strings for simulated data using lm, lme4, asreml, and sommer packages.
#' The models are based on the structure of the simulated data, which includes factors like year, location, treatment, genotype, and block.
generate_model_strings_for_simulated_data <- function(exclude_lm = FALSE, exclude_sommer = FALSE) {
  if (exclude_lm) {
    lm_model_strings <- c()
  } else {
    lm_model_strings <- c(
      "lm(y ~ year + loc + trt + gen + blk, data=df)",
      "lm(y ~ year * loc + trt + gen + blk, data=df)",
      "lm(y ~ env + gen + blk, data=df)"
    )
  }
  lmer_model_strings <- c(
    "lmer(y ~ year + loc + trt + blk + (1|gen), df)",
    "lmer(y ~ year * loc + trt + blk + (1|gen), df)",
    "lmer(y ~ year + loc + trt + blk + (1|gen) + (1|gen:year) + (1|gen:loc), df)",
    "lmer(y ~ env + blk + (1|gen), df)",
    "lmer(y ~ env + blk + (1|gen) + (1|gen:env), df)"
  )
  asreml_model_strings <- c(
    "asreml(y ~ year + loc + trt + blk, random = ~ gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc * trt + blk, random = ~ gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(loc):gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(year):gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(loc):gen + diag(year):gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + fa(loc):gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + fa(year):gen, data = df, trace = FALSE)",
    "asreml(y ~ year * loc * trt + blk, random = ~ gen + fa(year:loc):gen, data = df, trace = FALSE)",
    "asreml(y ~ year + loc + trt + blk, random = ~ gen + gen:year + gen:loc, data = df, trace = FALSE)",
    "asreml(y ~ env + blk, random = ~ gen, data = df, trace = FALSE)",
    "asreml(y ~ env + blk, random = ~ gen + fa(env):gen, data = df, trace = FALSE)"
  )
  if (exclude_sommer) {
    sommer_model_strings <- c()
  } else {
    sommer_model_strings <- c(
      "mmes(y ~ year + loc + trt + blk, random = ~ gen, data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen, data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(dsm(loc), ism(gen)), data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(usm(loc), ism(gen)), data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(dsm(year), ism(gen)), data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(usm(year), ism(gen)), data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(dsm(loc), ism(gen)) + vsm(dsm(year), ism(gen)), data = df)",
      "mmes(y ~ year + loc + trt + blk, random = ~ gen + vsm(usm(loc), ism(gen)) + vsm(usm(year), ism(gen)), data = df)",
      "mmes(y ~ env + blk, random = ~ gen + vsm(dsm(env), ism(gen)), data = df)",
      "mmes(y ~ env + blk, random = ~ gen + vsm(usm(env), ism(gen)), data = df)"
    )
  }
  model_strings <- if (nzchar(system.file(package = "asreml")) && nzchar(system.file(package = "sommer"))) {
    c(lm_model_strings, lmer_model_strings, asreml_model_strings, sommer_model_strings)
  } else if (nzchar(system.file(package = "asreml"))) {
    c(lm_model_strings, lmer_model_strings, asreml_model_strings)
  } else if (nzchar(system.file(package = "sommer"))) {
    c(lm_model_strings, lmer_model_strings, sommer_model_strings)
  } else {
    c(lm_model_strings, lmer_model_strings)
  }
  model_strings
}

#' Generate model strings for empirical data using lm, lme4, asreml, and sommer packages based on feature names.
#' On the other hand, these models are more data-driven and agnostic to the underlying data generating process, and are based on the feature names in the empirical data.
generate_model_strings_for_empirical_data <- function(x_names_except_gen_and_dummy_env, exclude_lm = FALSE, exclude_sommer = FALSE) {
  m <- length(x_names_except_gen_and_dummy_env)
  if (exclude_lm) {
    lm_model_strings <- c()
  } else {
    lm_model_strings <- c(
      "lm(y ~ gen, data=df)",
      "lm(y ~ dummy_env + gen, data=df)",
      "lm(y ~ dummy_env*gen, data=df)",
      paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + gen, data=df)"),
      paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env + gen, data=df)")
      # paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env*gen, data=df)"),
      # unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lm(y ~ ", x, " + gen, data=df)")})),
      # unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lm(y ~ ", x, "*gen, data=df)")}))
    )
    # if (m > 1) {
    #   for (i in 1:(m - 1)) {
    #     x1 <- x_names_except_gen_and_dummy_env[i]
    #     for (j in (i + 1):m) {
    #       x2 <- x_names_except_gen_and_dummy_env[j]
    #       lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen, data=df)"))
    #       lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen + ", x1, ":gen, data=df)"))
    #       lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, "*", x2, " + gen, data=df)"))
    #     }
    #   }
    # }
  }
  if (exclude_sommer) {
    lmer_model_strings <- c()
  } else {
    lmer_model_strings <- c(
      "lmer(y ~ (1|gen), data=df)",
      "lmer(y ~ dummy_env + (1|gen), data=df)",
      "lmer(y ~ dummy_env + (1|gen:dummy_env), data=df)",
      "lmer(y ~ (1|gen:dummy_env), data=df)",
      "lmer(y ~ (1|dummy_env) + (1|gen:dummy_env), data=df)",
      paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + (1|gen), data=df)"),
      paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env + (1|gen), data=df)"),
      paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + (1|gen) + (1|gen:dummy_env), data=df)"),
      paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env + (1|gen) + (1|gen:dummy_env), data=df)"),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lmer(y ~ ", x, " + (1|gen), data=df)")})),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lmer(y ~ ", x, " + (1|gen) + (1|gen:", x, "), data=df)")})),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lmer(y ~ (1|gen) + (1|gen:", x, "), data=df)")}))
    )
    if (m > 1) {
      for (i in 1:(m - 1)) {
        x1 <- x_names_except_gen_and_dummy_env[i]
        for (j in (i + 1):m) {
          x2 <- x_names_except_gen_and_dummy_env[j]
          lmer_model_strings <- c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen), data=df)"))
          lmer_model_strings <- c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen) + (1|gen:", x1, "), data=df)"))
          lmer_model_strings <- c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen) + (1|gen:", x2, "), data=df)"))
          lmer_model_strings <- c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen) + (1|gen:", x1, ") + (1|gen:", x2, "), data=df)"))
        }
      }
    }
  }
  asreml_model_strings <- c(
    "asreml(y ~ 1, random = ~ gen, data=df, trace=FALSE, maxit=10)",
    "asreml(y ~ dummy_env, random = ~ gen, data=df, trace=FALSE)",
    "asreml(y ~ dummy_env, random = ~ gen:dummy_env, data=df, trace=FALSE)",
    "asreml(y ~ 1, random = ~ gen:dummy_env, data=df, trace=FALSE)",
    "asreml(y ~ 1, random = ~ dummy_env + gen:dummy_env, data=df, trace=FALSE)",
    paste0("asreml(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), ", random = ~ gen, data=df, trace=FALSE)"),
    paste0("asreml(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), ", random = ~ gen + fa(dummy_env):gen, data=df, trace=FALSE)"),
    unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("asreml(y ~ ", x, ", random = ~ gen, data=df, trace=FALSE)")})),
    unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("asreml(y ~ ", x, ", random = ~ gen + ", x, ":gen, data=df, trace=FALSE)")})),
    unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("asreml(y ~ ", x, ", random = ~ gen + fa(", x, "):gen, data=df, trace=FALSE)")}))
  )
  if (m > 1) {
    for (i in 1:(m - 1)) {
      x1 <- x_names_except_gen_and_dummy_env[i]
      for (j in (i + 1):m) {
        x2 <- x_names_except_gen_and_dummy_env[j]
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x2, ":gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen + ", x2, ":gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + fa(", x1, "):gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + fa(", x2, "):gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + fa(", x1, "):gen + ", x2, ":gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen + fa(", x2, "):gen, data=df, trace=FALSE)"))
        asreml_model_strings <- c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen + fa(", x1, "):gen + fa(", x2, "):gen, data=df, trace=FALSE)"))
      }
    }
  }
  if (exclude_sommer) {
    sommer_model_strings <- c()
  } else {
    sommer_model_strings <- c(
      "mmes(y ~ 1, random = ~ gen, data=df, verbose=FALSE)",
      "mmes(y ~ dummy_env, random = ~ gen, data=df, verbose=FALSE)",
      "mmes(y ~ dummy_env, random = ~ gen + gen:dummy_env, data=df, verbose=FALSE)",
      "mmes(y ~ 1, random = ~ gen + gen:dummy_env, data=df, verbose=FALSE)",
      "mmes(y ~ 1, random = ~ dummy_env + gen + gen:dummy_env, data=df, verbose=FALSE)",
      paste0("mmes(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), ", random = ~ gen, data=df, verbose=FALSE)"),
      paste0("mmes(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), ", random = ~ gen + vsm(dsm(dummy_env), ism(gen)), data=df, verbose=FALSE)"),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("mmes(y ~ ", x, ", random = ~ gen, data=df, verbose=FALSE)")})),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("mmes(y ~ ", x, ", random = ~ gen + ", x, ":gen, data=df, verbose=FALSE)")})),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("mmes(y ~ ", x, ", random = ~ gen + vsm(dsm(", x, "), ism(gen)), data=df, verbose=FALSE)")}))
    )
    if (m > 1) {
      for (i in 1:(m - 1)) {
        x1 <- x_names_except_gen_and_dummy_env[i]
        for (j in (i + 1):m) {
          x2 <- x_names_except_gen_and_dummy_env[j]
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen, data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen, data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x2, ":gen, data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen + ", x2, ":gen, data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + vsm(dsm(", x1, "), ism(gen)), data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + vsm(dsm(", x2, "), ism(gen)), data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + vsm(dsm(", x1, "), ism(gen)) + ", x2, ":gen, data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + ", x1, ":gen + vsm(dsm(", x2, "), ism(gen)), data=df, verbose=FALSE)"))
          sommer_model_strings <- c(sommer_model_strings, paste0("mmes(y ~ ", x1, " + ", x2, ", random = ~ gen + vsm(dsm(", x1, "), ism(gen)) + vsm(dsm(", x2, "), ism(gen)), data=df, verbose=FALSE)"))
        }
      }
    }
  }
  model_strings <- if (nzchar(system.file(package = "asreml")) && nzchar(system.file(package = "sommer"))) {
    c(lm_model_strings, lmer_model_strings, asreml_model_strings, sommer_model_strings)
  } else if (nzchar(system.file(package = "asreml"))) {
    c(lm_model_strings, lmer_model_strings, asreml_model_strings)
  } else if (nzchar(system.file(package = "sommer"))) {
    c(lm_model_strings, lmer_model_strings, sommer_model_strings)
  } else {
    c(lm_model_strings, lmer_model_strings)
  }
  model_strings
}

#' Fit models using provided strings, select the best based on AIC/BIC/logLik, and extract genotype effects.
fit_extract_effects <- function(df, model_strings, time_limit_seconds = 1, verbose = TRUE) {
  # df <- process_features(df = read.table(list.files(path = ".", pattern = "simulated|.tsv")[1], header=TRUE))$df; time_limit_seconds = 1; verbose = TRUE;
  # x_names <- colnames(df)[2:ncol(df)]; x_names_except_gen_and_dummy_env <- x_names[(x_names != "gen") & (x_names != "dummy_env")]; model_strings <- generate_model_strings_for_empirical_data(x_names_except_gen_and_dummy_env, exclude_lm = TRUE, exclude_sommer = TRUE)
  model_candidates <- list()
  for (i in seq_along(model_strings)) {
    # i <- 1
    mod_string <- model_strings[i]
    mod_label <- unlist(strsplit(mod_string, "\\("))[1]
    if (verbose) {
      print(paste0("Fitting ", mod_label, "_", i, ": `", mod_string, "`"))
    }
    mod <- tryCatch(
      {
        setTimeLimit(elapsed = time_limit_seconds, transient = TRUE)
        # on.exit(setTimeLimit(elapsed = Inf), add = TRUE)
        # setTimeLimit(cpu = time_limit_seconds, elapsed = time_limit_seconds, transient = TRUE)
        eval(parse(text = mod_string))
        # withTimeout(eval(parse(text = mod_string)), timeout = time_limit_seconds)
      },
      error = function(e) {
        print(paste0("SKIPPED | Unable to fit: ", mod_string))
        NA
      }
    )
    if ((length(mod) == 1) && is.na(mod)) {
      model_candidates[[paste0(mod_label, "_", i)]] <- NA
    } else {
      if (class(mod) == "lmerMod") {
        if (mod@optinfo$conv$opt == 0) {
          # Failed to converge
          model_candidates[[paste0(mod_label, "_", i)]] <- NA
        } else {
          model_candidates[[paste0(mod_label, "_", i)]] <- mod
        }
      } else if (class(mod) == "asreml") {
        if (mod$converge == FALSE) {
          model_candidates[[paste0(mod_label, "_", i)]] <- NA
        } else {
          model_candidates[[paste0(mod_label, "_", i)]] <- mod
        }
      } else {
        model_candidates[[paste0(mod_label, "_", i)]] <- mod
      }
    }
  }
  df_stats <- data.frame(
    model = names(model_candidates),
    formula = model_strings,
    AIC = sapply(model_candidates, aic_lm_lmer_asreml),
    BIC = sapply(model_candidates, bic_lm_lmer_asreml),
    logLik = sapply(model_candidates, loglik_lm_lmer_asreml)
  )
  idx_filter <- which(!is.na(df_stats$AIC) & is.finite(df_stats$AIC))
  if (length(idx_filter) == 0) {
    print("NO MODEL WAS SUCCESSFULLY FITTED!")
    return(NULL)
  }
  if (length(idx_filter) == 0) {
    print("NO MODEL WAS SUCCESSFULLY FITTED!")
    return(NULL)
  }
  df_stats <- df_stats[idx_filter, ]
  model_candidates <- model_candidates[idx_filter]
  z_AIC <- scale(df_stats$AIC, scale = TRUE, center = TRUE)
  z_BIC <- scale(df_stats$BIC, scale = TRUE, center = TRUE)
  z_logLik <- -scale(df_stats$logLik, scale = TRUE, center = TRUE)
  df_stats$z_sum <- 0.2 * z_AIC + 0.6 * z_BIC + 0.2 * z_logLik # more weight on BIC because it is better for model fit parsimony rather than predictive accuracy which AIC is better suited for
  # Select the best model based on z_sum
  best_model_idx <- if (is.nan(df_stats$z_sum[1])) {
    # For non-varying stats or only a single model is left
    1
  } else {
    which.min(df_stats$z_sum)
  }
  # print(df_stats)
  # print(best_model_idx)
  best_model <- model_candidates[[best_model_idx]]
  best_model_formula <- df_stats$formula[best_model_idx]
  if (verbose) {
    print(df_stats)
    print(paste("Best model selected:", best_model_formula))
  }
  # Plot gen effects (random effects for gen)
  df_effects <- if (class(best_model) == "lm") {
    effects <- coef(best_model)
    ids <- names(effects)
    intercept <- effects[ids == "(Intercept)"]
    gen_effects <- c(intercept, intercept + effects[grepl("gen", ids)])
    gen_names <- c(
      as.character(levels(df$gen)[1]),
      ids[grepl("gen", ids)]
    )
    gen_names <- gsub("^gen", "", gen_names)
    data.frame(ids = gen_names, effects = gen_effects)
  } else if (class(best_model) == "lmerMod") {
    gen_effects <- ranef(best_model)$gen
    data.frame(ids = rownames(gen_effects), effects = gen_effects[, 1])
  } else if (class(best_model) == "asreml") {
    df_effects_temp <- data.frame(
      ids = rownames(coef(best_model)$random),
      effects = as.vector(coef(best_model)$random)
    )
    df_sub <- df_effects_temp[grepl("gen", df_effects_temp$ids) &

              !grepl(":", df_effects_temp$ids), ]
    df_sub$ids <- gsub("^gen_", "", df_sub$ids)
    df_sub
  } else if (class(best_model) == "mmes") {
    df_effects = if (length(best_model$uPevList) == 1) {
      X = best_model$u
      data.frame(ids=rownames(X), effects=X[, 1])
    } else {
      ids <- c()
      effects <- c()
      for (i in seq_along(best_model$uPevList)) {
        # i <- 1
        # best_model = model_candidates[[23]]
        bool_gen <- grepl("gen", names(best_model$uPevList)[i])
        bool_env <- (
          grepl("year", names(best_model$uPevList)[i]) |
          grepl("loc", names(best_model$uPevList)[i]) |
          grepl("trt", names(best_model$uPevList)[i]) |
          grepl("blk", names(best_model$uPevList)[i]) |
          grepl("env", names(best_model$uPevList)[i])
        )
        X <- best_model$uPevList[[i]]
        if (bool_gen && !bool_env) {
          ids = c(ids, rownames(X))
          effects = c(effects, X[, 1])
        } else if (bool_gen && bool_env) {
          for (j in 1:ncol(X)) {
            # j <- 1
            ids <- c(ids, paste0(colnames(X)[j], "▓", rownames(X)))
            effects <- c(effects, X[, j])
          }
        } else {
          next
        }
      }
      data.frame(ids, effects)
    }
    rownames(df_effects) <- NULL
    df_effects
  } else {
    data.frame()
  }
  # Add the expected delimiters for these "marginal" effects
  df_effects$ids <- gsub("_level", "➵level", df_effects$ids)
  df_effects$ids <- gsub(":", "▓", df_effects$ids)
  # Sort sensibly
  df_effects <- df_effects[stringr::str_order(df_effects$ids, numeric = TRUE), ]
  list(df_effects = df_effects, formula = best_model_formula)
}

#' Fit linear models and extract the genotype effects for simulated data by generating model strings and calling fit_extract_effects.
fit_extract_effects_for_simulated_data <- function(df, time_limit_seconds = 1, exclude_lm = FALSE, exclude_sommer = FALSE, verbose = TRUE) {
  # df = process_features(read.table(list.files(path = ".", pattern = ".tsv")[1], T))[[1]]; time_limit_seconds=1; exclude_lm=TRUE; exclude_sommer=TRUE; verbose=TRUE
  model_strings <- generate_model_strings_for_simulated_data(exclude_lm = exclude_lm, exclude_sommer = exclude_sommer)
  fit_extract_effects(df, model_strings, time_limit_seconds = time_limit_seconds, verbose = verbose)
}

#' Fit linear models and extract the genotype effects for empirical data by generating model strings based on features and calling fit_extract_effects.
fit_extract_effects_for_empirical_data <- function(df, time_limit_seconds = 1, exclude_lm = FALSE, exclude_sommer = FALSE, verbose = TRUE) {
  # df = process_features(read.table(list.files(path = ".", pattern = ".tsv")[19], T))[[1]]; time_limit_seconds=1; exclude_lm=TRUE; exclude_sommer=TRUE; verbose=TRUE
  x_names <- colnames(df)[2:ncol(df)]
  x_names_except_gen_and_dummy_env <- x_names[(x_names != "gen") & (x_names != "dummy_env")]
  model_strings <- generate_model_strings_for_empirical_data(x_names_except_gen_and_dummy_env, exclude_lm = exclude_lm, exclude_sommer = exclude_sommer)
  fit_extract_effects(df, model_strings, time_limit_seconds = time_limit_seconds, verbose = verbose)
}

#' Compute cross-validation metrics (Pearson correlation, MAE, MSE, RMSE, R-squared) between predicted and observed values.
cv_metrics <- function(yHat, y) {
  # yHat = rnorm(100); y = rnorm(100)
  n <- length(y)
  pcor <- cor(yHat, y)
  mae <- mean(abs(yHat - y))
  mse <- mean((yHat - y)^2)
  rmse <- sqrt(mse)
  r2 <- 1.00 - (sum((yHat - y)^2) / sum((y - mean(y))^2))
  list(
    pcor = pcor,
    mae = mae,
    mse = mse,
    rmse = rmse,
    r2 = r2
  )
}

#' Given an input filename, generate the corresponding output filename by appending "output-", and changing the extension to "-LINEAR.tsv".
define_fname_output <- function(fname_input) {
  dirname_input <- dirname(fname_input)
  basename_input <- basename(fname_input)
  fname_output <- paste0("output-", gsub(".tsv", "-LINEAR.tsv", basename_input))
  file.path(dirname_input, fname_output)
}

#' Main function to extract genotype effects from either simulated or empirical data by fitting various linear models and selecting the best one based on model selection criteria, then saving the results to an output file.
extract_entries_effects <- function(params) {
  # params = get_params(args=c("trials", "australia.soybean-yield.tsv", "FALSE", "TRUE", "TRUE", "TRUE"))
  input_list <- process_features(df = read.table(params$fname_input, sep = "\t", header = TRUE))
  df <- input_list$df
  attach(df)
  out <- if (params$trials$is_simulated) {
    fit_extract_effects_for_simulated_data(
      df,
      exclude_lm = params$trials$exclude_lm,
      exclude_sommer = params$trials$exclude_sommer,
      verbose = params$trials$verbose
    )
  } else {
    fit_extract_effects_for_empirical_data(
      df,
      exclude_lm = params$trials$exclude_lm,
      exclude_sommer = params$trials$exclude_sommer,
      verbose = params$trials$verbose
    )
  }
  if (is.null(out)) {
    out = list(df_effects = data.frame(ids = character(), effects = numeric()), formula = NA)
  }
  fname_output <- define_fname_output(params$fname_input)
  write.table(out$df_effects, file = fname_output, row.names = FALSE, col.names = TRUE,  sep = "\t")
  fname_output
}

#' Main function to perform repeated k-fold cross-validation using Bayesian genomic prediction models (e.g., BRR, BayesA, BayesB, BayesC) on a given dataset, and save the results (correlation and R-squared) for each fold and repetition to an output file.
gp_repeated_kfold_cv <- function(params) {
  # params = get_params(args=c("gp", "sorghum-YLD.tsv", "2", "1", "100", "10"))
  fname_output <- define_fname_output(params$fname_input)
  fname_output_tmp <- paste0(fname_output, ".tmp")
  cat(paste(c("datasets", "reps", "folds", "nt", "nv", "models", "corr", "r2"), collapse = "\t"), file = fname_output_tmp, sep = "\n")
  df <- read.table(params$fname_input, sep = "\t", header = TRUE)
  df < df[complete.cases(df), ]
  n <- nrow(df)
  p <- ncol(df) - 1
  m <- floor(n / params$gp$n_folds)
  if (m < 3) {
    stop(paste0("ERROR: Skipping because the dataset (", params$fname_input, ") is too small (n=", n, "; m=", m, ") for ", params$gp$n_reps, "-reps of ", params$gp$n_folds, "-fold cross-validation"))
  }
  y <- df[, 1, drop = FALSE]
  X <- df[, 2:ncol(df), drop = FALSE]
  for (r in 1:params$gp$n_reps) {
    # r <- 1
    set.seed(params$gp$base_seed + r)
    idx_shuffled <- sample(1:n, n, replace = FALSE)
    for (f in 1:params$gp$n_folds) {
      # f <- 2
      bool_validation <- (1:n) %in% (((f-1)*m)+1):(f*m)
      bool_training <- !bool_validation
      idx_training <- idx_shuffled[bool_training]
      idx_validation <- idx_shuffled[bool_validation]
      yNA <- y[, 1]
      yNA[idx_validation] <- NA
      for (model in params$gp$models) {
        # model = "BayesC"
        mod <- BGLR(
          y = yNA,
          ETA = list(list(X = X, model = model)),
          nIter = params$gp$n_iterations,
          burnIn = params$gp$n_burnin_iterations,
          saveAt = paste0(params$fname_input, "-", model, "-"),
          verbose = params$gp$verbose
        )
        yHat <- mod$yHat[idx_validation]
        res <- cv_metrics(yHat, y[idx_validation, ])
        data <- paste(c(basename(params$fname_input), r, f, length(idx_training), length(idx_validation), model, res$pcor, res$r2), collapse = "\t")
        cat(data, file = fname_output_tmp, sep = "\n", append = TRUE)
        unlink(paste0(params$fname_input, "-", model, "-*"))
      }
    }
  }
  file.rename(from = fname_output_tmp, to = fname_output)
  fname_output
}

#' Simulate a dataset with specific structure and save it to a TSV file for testing the modeling functions.
misc_sim <- function() {
  df <- expand.grid(
    year = c("Year➵2026", "Year➵2027"),
    loc = c("Loc➵A", "Loc➵B"),
    trt = c("Trt➵1", "Trt➵2"),
    blk = c("Blk➵1", "Blk➵2", "Blk➵3"),
    gen = paste0("Gen➵", 1:100)
  )
  df$y <- rnorm(nrow(df))
  # We need to have the response variable (y) as the first column for the modeling functions to work correctly, so we reorder the columns accordingly.
  df <- df[, c("y", "year", "loc", "trt", "blk", "gen")]
  write.table(df, file = "simulated_misc.tsv", row.names = FALSE, col.names = TRUE, quote = FALSE, sep = "\t")
}

###########################################################
# Execute
###########################################################
# Testing: source("scripts/linear.R")
# misc_sim()
# args <- c("trials", "simulated_misc.tsv", "TRUE", "TRUE", "TRUE", "TRUE")
# args <- c("trials", "australia.soybean-yield.tsv", "FALSE", "TRUE", "TRUE", "TRUE")
# args <- c("gp", "sorghum-YLD.tsv", "2", "1", "100", "10", "BRR,BayesA", "TRUE", "42")
# args <- c("trials", "/home/jp3h/Documents/mlp/tests/tmp/trials/ilri.sheep-birthwt.tsv", "/home/jp3h/Documents/mlp/tests/tmp/trials", "FALSE", "FALSE", "TRUE", "TRUE")
params <- get_params(args)
if (params$analysis_type == "trials") {
  extract_entries_effects(params)
} else {
  gp_repeated_kfold_cv(params)
}
