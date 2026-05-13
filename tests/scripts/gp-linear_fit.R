library(BGLR)
# library(doParallel)
args = commandArgs(trailingOnly=TRUE)
n_folds <- 10
n_reps <- 5
nIter <- 1500
burnIn <- 500
cv_metrics <- function(yHat, y) {
  # yHat = rnorm(100); y = rnorm(100)
  n = length(y)
  pcor = cor(yHat, y)
  mae = mean(abs(yHat - y))
  mse = mean((yHat - y)^2)
  rmse = sqrt(mse)
  r2 = 1.00 - (sum((yHat - y)^2) / sum((y - mean(y))^2))
  list(
    pcor = pcor,
    mae = mae,
    mse = mse,
    rmse = rmse,
    r2 = r2
  )
}
fit <- function(fname_input) {
  # fname_input <- list.files(path = ".", pattern = "^input_simulated-.*.tsv")[1]
  # fname_input <- list.files(path = ".", pattern = "*.*.tsv")[1]
  fname_output = gsub("input_simulated", "output_simulated", gsub(".tsv", "-LINEAR.tsv", fname_input))
  fname_output = if (grepl("^output", basename(fname_output))) {
    fname_output
  } else {
    paste0(dirname(fname_output), "/output-", basename(fname_output))
  }
  cat(paste(c("datasets", "reps", "folds", "nt", "nv", "models", "corr", "r2"), collapse="\t"), file=fname_output, sep="\n")
  datasets = c()
  reps = c()
  folds = c()
  nt = c()
  nv = c()
  models = c()
  corr = c()
  r2 = c()
  df <- read.table(fname_input, sep = "\t", header = TRUE)
  df < df[complete.cases(df), ]
  n <- nrow(df)
  p <- ncol(df) - 1
  m <- ceiling(n / n_folds)
  if (m < 3) {
    print(paste0("ERROR: Skipping because the dataset (", fname_input, ") is too small (n=", n, "; m=", m, ") for ", n_reps, "-reps of ", n_folds, "-fold cross-validation"))
    return(1)
  }
  y <- df[, 1, drop = FALSE]
  X <- df[, 2:ncol(df), drop = FALSE]
  for (r in 1:n_reps) {
    # r <- 1
    set.seed(r)
    idx_shuffled <- sample(1:n, n, replace = FALSE)
    for (f in 1:n_folds) {
      # f <- 2
      bool_validation <- (1:n) %in% (((f-1)*m)+1):(f*m)
      bool_training <- !bool_validation
      idx_training <- idx_shuffled[bool_training]
      idx_validation <- idx_shuffled[bool_validation]
      yNA <- y[, 1]
      yNA[idx_validation] <- NA
      for (model in c("BRR", "BayesA", "BayesB", "BayesC")) {
        # model = "BayesC"
        mod <- BGLR(y = yNA, ETA = list(list(X = X, model = model)), nIter = nIter, burnIn = burnIn, saveAt = paste0(fname_input, "-", model, "-"), verbose=TRUE)
        yHat <- mod$yHat[idx_validation]
        res = cv_metrics(yHat, y[idx_validation, ])
        datasets = c(datasets, fname_input)
        reps = c(reps, r)
        folds = c(folds, f)
        nt = c(nt, length(idx_training))
        nv = c(nv, length(idx_validation))
        models = c(models, model)
        corr = c(corr, res$pcor)
        r2 = c(r2, res$r2)
        data = paste(c(fname_input, r, f, length(idx_training), length(idx_validation), model, res$pcor, res$r2), collapse="\t")
        cat(data, file = fname_output, sep = "\n", append=TRUE)
        unlink(paste0(args[1], "-", model, "-*"))
      }
    }
  }
  data.frame(datasets = datasets, reps = reps, folds = folds, nt = nt, nv = nv, models = models, corr = corr, r2 = r2)
}
# # Run on parallel
# fnames_tmp <- list.files(path = ".", pattern = "^input_simulated-.*.tsv")
# fnames <- if (length(fnames_tmp) > 0) {
#   fnames_tmp
# } else {
#   list.files(path = ".", pattern = "[maize|rice|sorghum|soy|spruce|switchgrass]-.*.tsv")
# }``
# system.time({
#     mclapply(fnames, fit, mc.cores = length(fnames))
# })
fit(fname_input = args[1])
