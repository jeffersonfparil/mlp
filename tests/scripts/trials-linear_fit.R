library("R.utils")
library("stringr")
library("lme4")
if (nzchar(system.file(package = "asreml"))) {
  library("asreml") # requires ```shell module load ASReml-R ```
}
if (nzchar(system.file(package = "sommer"))) {
  library("sommer") # Too slow for non-trially large datasets, e.g. 21,000 observations simulated here
}

#' Process features in the dataframe by converting explanatory variables to factors and creating a dummy environment variable if needed.
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
generate_model_strings_for_simulated_data <- function(exclude_sommer = FALSE) {
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
      paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env + gen, data=df)"),
      paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse = ' + ' ), " + dummy_env*gen, data=df)"),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lm(y ~ ", x, " + gen, data=df)")})),
      unlist(lapply(x_names_except_gen_and_dummy_env, FUN = function(x) {paste0("lm(y ~ ", x, "*gen, data=df)")}))
    )
    if (m > 1) {
      for (i in 1:(m - 1)) {
        x1 <- x_names_except_gen_and_dummy_env[i]
        for (j in (i + 1):m) {
          x2 <- x_names_except_gen_and_dummy_env[j]
          lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen, data=df)"))
          lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen + ", x1, ":gen, data=df)"))
          lm_model_strings <- c(lm_model_strings, paste0("lm(y ~ ", x1, "*", x2, " + gen, data=df)"))
        }
      }
    }
  }
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
  model_candidates <- list()
  for (i in seq_along(model_strings)) {
    # i <- 30
    mod_string <- model_strings[i]
    mod_label <- unlist(strsplit(mod_string, "\\("))[1]
    if (verbose) {
      print(paste0("Fitting ", mod_label, "_", i, ": `", mod_string, "`"))
    }
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
  idx_filter = which(!is.na(df_stats$AIC) & is.finite(df_stats$AIC))
  df_stats = df_stats[idx_filter, ]
  model_candidates = model_candidates[idx_filter]
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

#' Fit linear models and extract the genotype effects for simulated data by generating model strings and calling fit_extract_effects.
fit_extract_effects_for_simulated_data <- function(df, time_limit_seconds = 1, exclude_lm = FALSE, exclude_sommer = FALSE, verbose = TRUE) {
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

#' Run the analysis on all input files in the directory, processing simulated or empirical data and outputting results.
run <- function(exclude_lm = FALSE, exclude_sommer = FALSE) {
  fnames_tmp <- list.files(path = ".", pattern = "input_simulated")
  fnames <- if (length(fnames_tmp) != 0) {
    fnames_tmp
  } else {
    list.files(path = ".", pattern = ".tsv")
  }
  output <- list()
  for (fname_input in fnames) {
    # fname_input <- fnames[19]
    input_list <- process_features(df = read.table(fname_input, header=TRUE))
    df <- input_list$df
    # str(df)
    # ids_features <- input_list$ids_features
    attach(df)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(fname_input)
    out <- if (length(fnames_tmp) != 0) {
      fit_extract_effects_for_simulated_data(df, exclude_lm = exclude_lm, exclude_sommer = exclude_sommer, verbose=FALSE)
    } else {
      fit_extract_effects_for_empirical_data(df, exclude_lm = exclude_lm, exclude_sommer = exclude_sommer, verbose=FALSE)
    }
    fname_output <- if (length(fnames_tmp) != 0) {
      paste0(
        gsub("^input", "output", gsub(".tsv", "", fname_input)),
        "-LINEAR_",
        gsub(" ", "", out$formula),
        ".tsv"
      )
    } else {
      paste0("output-", gsub(".tsv$", "", fname_input),
        "-LINEAR_",
        gsub(" ", "", out$formula),
        ".tsv"
      )
    }
    write.table(out$df_effects,
      file = fname_output, row.names = FALSE,
      col.names = TRUE, sep = "\t"
    )
    output[[fname_input]] <- out
    detach(df)
  }
  output
}

# Execute
run(exclude_lm = TRUE, exclude_sommer = TRUE)