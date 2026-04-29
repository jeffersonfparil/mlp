library("stringr")
library("lme4")
if (nzchar(system.file(package = "asreml"))) {
    library("asreml") # requires ```shell module load ASReml-R ```
}

process_features = function(df) {
    ids_features = c()
    for (j in 1:ncol(df)) {
        # j = 7
        if (is.character(df[, j])) {
            df[, j] = as.factor(df[, j])
            ids_features = c(ids_features, paste0(names(df)[j], sort(levels(df[, j]))))
        }
    }
    for (v in c("year", "site", "treatment", "entry", "block")) {
        if (!is.factor(df[[v]])) df[[v]] = as.factor(df[[v]])
    }
    return(list(
        df=df, 
        ids_features=ids_features
    ))
}

AIC_lm_lmer_asreml = function(mod) {
    # mod = model_candidates[[13]]
    if ((class(mod) == "lm") | (class(mod) == "lmerMod")) {
        return(AIC(mod))
    } else if (class(mod) == "asreml") {
        return(-2*mod$loglik + 2*nrow(summary(mod)$varcomp))
    } else {
        print("Unknown model class. We expect 'lm', 'lme4' or 'asreml'.")
        NA
    }
}
BIC_lm_lmer_asreml = function(mod) {
    # mod = model_candidates[[13]]
    if ((class(mod) == "lm") | (class(mod) == "lmerMod")) {
        return(BIC(mod))
    } else if (class(mod) == "asreml") {
        return(-2*mod$loglik + nrow(summary(mod)$varcomp)*log(summary(mod)$nedf))
    } else {
        print("Unknown model class. We expect 'lm', 'lme4' or 'asreml'.")
        NA
    }
}
logLik_lm_lmer_asreml = function(mod) {
    # mod = model_candidates[[1]]
    if ((class(mod) == "lm") | (class(mod) == "lmerMod")) {
        return(as.numeric(logLik(mod)))
    } else if (class(mod) == "asreml") {
        return(mod$loglik)
    } else {
        print("Unknown model class. We expect 'lm', 'lme4' or 'asreml'.")
        NA
    }
}
ndf_lm_lmer_asreml = function(mod) {
    # mod = model_candidates[[13]]
    if ((class(mod) == "lm") | (class(mod) == "lmerMod")) {
        return(attr(logLik(mod), "df"))
    } else if (class(mod) == "asreml") {
        return(summary(mod)$nedf)
    } else {
        print("Unknown model class. We expect 'lm', 'lme4' or 'asreml'.")
        NA
    }
}

fit_extract_effects = function(df) {
    lm_model_strings = c(
        "lm(y ~ year + site + treatment + entry + block, data=df)",
        "lm(y ~ year * site + treatment + entry + block, data=df)"
    )
    lmer_model_strings = c(
        'lmer(y ~ year + site + treatment + block + (1|entry), df)',
        'lmer(y ~ year * site + treatment + block + (1|entry), df)',
        # 'lmer(y ~ year * site * treatment + block + (1|entry), df)',
        # 'lmer(y ~ year * site + treatment + block + (1 + year|entry), df)',
        'lmer(y ~ year + site + treatment + block + (1|entry) + (1|entry:year) + (1|entry:site), df)'
    )
    asreml_model_strings = c(
        'asreml(y ~ year + site + treatment + block, random = ~ entry, data = df)',
        'asreml(y ~ year * site + treatment + block, random = ~ entry, data = df)',
        'asreml(y ~ year * site * treatment + block, random = ~ entry, data = df)',
        'asreml(y ~ year * site + treatment + block, random = ~ entry + fa(site):entry, data = df)',
        'asreml(y ~ year * site + treatment + block, random = ~ entry + fa(year):entry, data = df)',
        'asreml(y ~ year * site * treatment + block, random = ~ entry + fa(year:site):entry, data = df)',
        'asreml(y ~ year + site + treatment + block, random = ~ entry + entry:year + entry:site, data = df)'
    )
    model_strings = if (nzchar(system.file(package = "asreml"))) {
        c(lm_model_strings, lmer_model_strings, asreml_model_strings)
    } else {
        c(lm_model_strings, lmer_model_strings)
    }
    # Fit these models
    model_candidates = list()
    for (i in 1:length(model_strings)) {
        # i = 13
        # i = length(model_strings)
        mod_string = model_strings[i]
        mod_label = unlist(strsplit(mod_string, "\\("))[1]
        print(paste0("Fitting ", mod_label, "_", i, ": `", mod_string, "`"))
        mod = tryCatch(
            eval(parse(text=mod_string)),
            error = function(e) {
                print("Unable to fit: skipped!")
                return(NA)
            }
        )
        if ((length(mod) == 1) && is.na(mod)) {
            model_candidates[[paste0(mod_label, "_", i)]] = NA
        } else {
            model_candidates[[paste0(mod_label, "_", i)]] = mod
        }
    }
    df_stats = data.frame(
        model = names(model_candidates),
        formula = model_strings,
        AIC = sapply(model_candidates, AIC_lm_lmer_asreml),
        BIC = sapply(model_candidates, BIC_lm_lmer_asreml),
        logLik = sapply(model_candidates, logLik_lm_lmer_asreml)
    )
    z_AIC = scale(df_stats$AIC, scale=T, center=T)
    z_BIC = scale(df_stats$BIC, scale=T, center=T)
    z_logLik = -scale(df_stats$logLik, scale=T, center=T)
    df_stats$z_sum = 0.2*z_AIC + 0.6*z_BIC + 0.2*z_logLik
    print(df_stats)
    # Select the best model based on z_sum
    # best_model_idx = which.min(df_stats$BIC)
    best_model_idx = which.min(df_stats$z_sum)
    best_model = model_candidates[[best_model_idx]]
    best_model_formula = df_stats$formula[best_model_idx]
    print(paste("Best model selected:", best_model_formula))

    # Plot entry effects (random effects for entry)
    # best_model = model_candidates[[1]]
    df_effects = if (class(best_model) == "lm") {
        # best_model = model_candidates[[1]]
        effects = coef(best_model)
        ids = names(effects)
        intercept = effects[ids == "(Intercept)"]
        entry_effects = c(intercept, intercept + effects[grepl("entry", ids)])
        entry_names = c(as.character(levels(df$entry)[1]), ids[grepl("entry", ids)])
        entry_names = gsub("entry", "", entry_names)
        df_effects = data.frame(ids=entry_names, effects=entry_effects)
        rownames(df_effects) = NULL
        df_effects
        # barplot(entry_effects, names.arg=entry_names, main = "Estimated Entry Effects (fixed effects model)", xlab = "Entry", ylab = "Coefficients")
    } else if (class(best_model) == "lmerMod") {
        # best_model = model_candidates[[3]]
        entry_effects <- ranef(best_model)$entry
        df_effects = data.frame(ids=rownames(entry_effects), effects=entry_effects[,1])
        rownames(df_effects) = NULL
        df_effects
        # barplot(entry_effects[,1], names.arg = rownames(entry_effects), main = "Estimated Entry Effects (mixed model)", xlab = "Entry", ylab = "Random Effect")
    } else if (class(best_model) == "asreml") {
        # best_model = model_candidates[[13]]
        df_effects = data.frame(
            ids = rownames(coef(best_model)$random),
            effects = as.vector(coef(best_model)$random)
        ); row.names(df_effects) = NULL
        # str(df_effects)
        df_sub = df_effects[grepl("entry", df_effects$ids) & !grepl(":", df_effects$ids), ]
        df_sub$ids = gsub("entry_", "", df_sub$ids)
        df_effects
        # barplot(df_sub$effects, names.arg = df_sub$ids, main = "Estimated Entry Effects (asreml model)", xlab = "Entry", ylab = "Random Effect")
    } else {
        data.frame()
        # plot(0, 0)
        print("Unknown model class. We expect 'lm', 'lme4' or 'asreml'.")
    }
    # Add the expected delimiters for these "marginal" effects
    df_effects$ids = gsub("_level", "➵level", df_effects$ids)
    df_effects$ids = gsub(":", "▓", df_effects$ids)
    # Sort sensibly
    df_effects = df_effects[stringr::str_order(df_effects$ids, numeric=TRUE), ]
    return(list(
        df_effects=df_effects,
        formula=best_model_formula
    ))
}

### Fit and extract entry effects
fnames = list.files(path=".", pattern="input_simulated")
output = list()
for (fname_input in fnames) {
    # fname_input = "input_simulated-NORMAL-1HL.tsv"
    input_list = process_features(df=read.delim(fname_input, T))
    df = input_list$df
    ids_features = input_list$ids_features
    attach(df)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(fname_input)
    out = fit_extract_effects(df)
    fname_output = paste0(
        gsub("^input", "output", gsub(".tsv", "", fname_input)),
        "-LINEAR_",
        gsub(" ", "", out$formula), 
        ".tsv"
    )
    write.table(out$df_effects, file=fname_output, row.names=FALSE, col.names=TRUE, sep="\t")
    output[[fname_input]] = out
    detach(df)
}
