library("R.utils")
library("stringr")
library("lme4")
if (nzchar(system.file(package = "asreml"))) {
  library("asreml") # requires ```shell module load ASReml-R ```
}
if (nzchar(system.file(package = "sommer"))) {
  library("sommer") # Too slow for non-trially large datasets, e.g. 21,000 observations simulated here
}

process_features <- function(df) {
  ids_features <- c()
  for (j in seq_len(ncol(df))) {
    # j <- 7
    if (is.character(df[, j])) {
      df[, j] <- as.factor(df[, j])
      ids_features <- c(
        ids_features,
        paste0(names(df)[j], sort(levels(df[, j])))
      )
    }
  }
  for (v in c("year", "loc", "trt", "gen", "blk")) {
    if (!is.factor(df[[v]])) {
      df[[v]] <- as.factor(df[[v]])
    }
  }
  # Include an env variable merging years, locs, and trts into 1 factor
  df$env <- paste0("year_", df$year, "|loc_", df$loc, "|trt_", df$trt)
  df$env <- as.factor(df$env)
  list(df = df, ids_features = ids_features)
}

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

generate_model_strings <- function() {
  lm_model_strings <- c(
    "lm(y ~ year + loc + trt + gen + blk, data=df)",
    "lm(y ~ year * loc + trt + gen + blk, data=df)",
    "lm(y ~ env + gen + blk, data=df)"
  )
  lmer_model_strings <- c(
    "lmer(y ~ year + loc + trt + blk + (1|gen), df)",
    "lmer(y ~ year * loc + trt + blk + (1|gen), df)",
    "lmer(y ~ year + loc + trt + blk + (1|gen) + (1|gen:year) + (1|gen:loc), df)",
    "lmer(y ~ env + blk + (1|gen), df)",
    "lmer(y ~ env + blk + (1|gen) + (1|gen:env), df)"
  )
  asreml_model_strings <- c(
    "asreml(y ~ year + loc + trt + blk, random = ~ gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen, data = df)",
    "asreml(y ~ year * loc * trt + blk, random = ~ gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(loc):gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(year):gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + diag(loc):gen + diag(year):gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + fa(loc):gen, data = df)",
    "asreml(y ~ year * loc + trt + blk, random = ~ gen + fa(year):gen, data = df)",
    "asreml(y ~ year * loc * trt + blk, random = ~ gen + fa(year:loc):gen, data = df)",
    "asreml(y ~ year + loc + trt + blk, random = ~ gen + gen:year + gen:loc, data = df)",
    "asreml(y ~ env + blk, random = ~ gen, data = df)",
    "asreml(y ~ env + blk, random = ~ gen + fa(env):gen, data = df)"
  )
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

fit_extract_effects <- function(df, time_limit_seconds=1, verbose=TRUE) {
  # time_limit_seconds=120; verbose=TRUE;
  # Generate model strings
  model_strings <- generate_model_strings()
  # Fit these models
  model_candidates <- list()
  for (i in seq_along(model_strings)) {
    # i <- 30
    mod_string <- model_strings[i]
    mod_label <- unlist(strsplit(mod_string, "\\("))[1]
    print(paste0("Fitting ", mod_label, "_", i, ": `", mod_string, "`"))
    mod <- tryCatch(
      {
        # setTimeLimit(time_limit_seconds, transient = TRUE)
        # setTimeLimit(cpu = time_limit_seconds, elapsed = time_limit_seconds, transient = TRUE)
        # eval(parse(text = mod_string))
        withTimeout(eval(parse(text = mod_string)), timeout = time_limit_seconds)
      },
      error = function(e) {
        print("Unable to fit: skipped!")
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
  z_AIC <- scale(df_stats$AIC, scale = TRUE, center = TRUE)
  z_BIC <- scale(df_stats$BIC, scale = TRUE, center = TRUE)
  z_logLik <- -scale(df_stats$logLik, scale = TRUE, center = TRUE)
  df_stats$z_sum <- 0.2 * z_AIC + 0.6 * z_BIC + 0.2 * z_logLik # more weight on BIC because it is better for model fit parsinomy rather than predictive accuracy which AIC is better suited for
  # Select the best model based on z_sum
  best_model_idx <- which.min(df_stats$z_sum)
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
    gen_names <- gsub("gen", "", gen_names)
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
    df_sub$ids <- gsub("gen_", "", df_sub$ids)
    df_sub
  } else if (class(best_model) == "mmes") {
    df_effects = if (length(best_model$uPevList) == 1) {
      X = best_model$u
      data.frame(ids=rownames(X), effects=X[, 1])
    } else {
      ids = c()
      effects = c()
      for (i in 1:length(best_model$uPevList)) {
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
        X = best_model$uPevList[[i]]
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
  return(list(df_effects = df_effects, formula = best_model_formula))
}

### Fit and extract gen effects
run <- function() {
  fnames <- list.files(path = ".", pattern = "input_simulated")
  output <- list()
  for (fname_input in fnames) {
    # fname_input <- fnames[6]
    input_list <- process_features(df = read.table(fname_input, header=TRUE))
    df <- input_list$df
    # ids_features <- input_list$ids_features
    attach(df)
    print(paste0(
      "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
      "@@@@@@"
    ))
    print(fname_input)
    out <- fit_extract_effects(df)
    fname_output <- paste0(
      gsub("^input", "output", gsub(".tsv", "", fname_input)),
      "-LINEAR_",
      gsub(" ", "", out$formula),
      ".tsv"
    )
    write.table(out$df_effects,
      file = fname_output, row.names = FALSE,
      col.names = TRUE, sep = "\t"
    )
    output[[fname_input]] <- out
    detach(df)
  }
  return(output)
}

run()