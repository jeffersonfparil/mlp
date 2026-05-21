args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
  stop("No file names provided as arguments.")
}

define_fname_output <- function(fname, y_name=NULL) {
  fname_out <- if (is.null(y_name)) {
    gsub(".txt$", ".tsv", fname)
  } else {
    gsub(".csv$", paste0("-", y_name, ".tsv"), fname_geno)
  }
  fname_out
}

prepare_trial_data <- function(fname) {
  # fname <- fnames[1]
  # fname <- "archbold.apple.txt"
  # fname <- "acorsi.grayleafspot.txt"
  # fname <- "alwan.lamb.txt"
  # fname <- "aastveit.barley.height.txt"
  if (grepl(".covs.txt", fname)) {
    stop("Covariate file found: ", fname)
  }
  if (grepl(".uniformity.txt", fname)) {
    stop("Uniformity file found: ", fname)
  }
  df <- read.table(fname, header = TRUE, na.strings = c("", "NA", "NAN", "NaN", "na", "nan"))
  potential_environmental_variable_names <- c(
    "year", "years",
    "loc", "locs",
    "harvest", "harvests",
    "season", "seasons",
    "plot", "plots",
    "rep", "reps",
    "row", "rows",
    "col", "cols",
    "blk", "blks",
    "replication", "replications",
    "column", "columns",
    "block", "blocks",
    "pos", "position", "positions",
    "spacing", "spacings",
    "stock", "stocks",
    "trt", "trts",
    "treatment", "treatments"
  )
  potential_explanatory_names <- c(
    "gen", "gens",
    "genotype", "genotypes",
    "entry", "entries",
    "pig", "pigs",
    "animal", "animals",
    "id", "ids",
    "breed", "breeds",
    "sire", "sires",
    "male", "males",
    "female", "females",
    "tree", "trees",
    "group", "groups",
    "zone", "zones",
    "isle", "isles",
    "sex",
    "pop", "pops",
    "var", "vars",
    "env", "envs",
    "genotype", "genotypes",
    "population", "populations",
    "variety", "varieties",
    "cultivar", "cultivars",
    potential_environmental_variable_names
  )
  potential_response_names <- c(
    "yield",
    "grain",
    "straw",
    "height",
    "size",
    "lodging",
    "protein",
    "oil"
  )
  too_specific_ids_when_gen_is_already_present <- c(
    "id", "ids",
    "plot", "plots",
    "entry", "entries",
    "entry_number"
  )
  some_specific_factor_names <- c(
    # barrero.maize
    "yor",
    # carlson.germination
    "nacl",
    # durban.splitplot
    "bed",
    # edwards.oats
    "eid",
    # gotway.hessianfly
    "lat",
    "long",
    # hanks.sprinkler
    "subplot",
    "irr",
    # ilri.sheep
    "ewe",
    "damage",
    "lamb",
    "ram",
    # kreusler.maize
    "rain",
    "temp",
    "raindays",
    "parentseed",
    # theobald.barley
    "nitro",
    # tesfaye.millet
    "entry_number",
    # verbyla.lupin
    "rate",
    "linrow",
    "lincol",
    "linrate"
  )
  explanatory_names <- c()
  response_names <- c()
  for (j in seq_len(ncol(df))) {
    # j <- 1
    id <- names(df)[j]
    y <- df[, j]
    n <- length(y)
    if (id %in% potential_explanatory_names) {
      if (is.numeric(y) && (length(unique(y)) > 5) && (var(y, na.rm = TRUE) > 1e-7) && !(id %in% potential_environmental_variable_names)) {
        response_names <- c(response_names, id)
      } else {
        explanatory_names <- c(explanatory_names, id)
        df[, j] = paste0(id, "➵", y)
      }
    } else {
      if (id %in% potential_response_names) {
        response_names <- c(response_names, id)
      } else {
        if (!is.character(y) && !((length(unique(y)) < 5) || (var(y, na.rm = TRUE) < 1e-7)) && !(id %in% some_specific_factor_names)) {
          response_names <- c(response_names, id)
        } else {
          explanatory_names <- c(explanatory_names, id)
          df[, j] = paste0(id, "➵", y)
        }
      }
    }
  }
  # Remove plot or location identifier which fixes each observation to one id
  idx_fitler_ids <- unlist(lapply(explanatory_names, FUN = function(x) {!(x %in% too_specific_ids_when_gen_is_already_present)}))
  explanatory_names <- explanatory_names[idx_fitler_ids]
  # Identify column coordinates of the explanatory variables
  idx_explanatories <- which(names(df) %in% explanatory_names)
  # print(colnames(df[idx_explanatories]))
  if (length(idx_explanatories) == 0) {
    next
  }
  # FOR SIMPLICITY WE ARE ONLY INCLUDING THOSE DATASETS WITH `gen` because we are ultimately interested in ranking genotypes for breeding purposes
  if (!("gen" %in% colnames(df)[idx_explanatories])) {
    next
  }
  if (max(unique(table(df$gen))) == 1) {
    next
  }
  df_explanatories <- df[, idx_explanatories, drop=FALSE]
  for (y_name in response_names) {
    # y_name = response_names[1]
    df_out <- cbind(data.frame(y = df[, which(names(df) == y_name)]), df_explanatories)
    df_out = df_out[complete.cases(df_out), ]
    if (nrow(df_out) < 2*length(unique(df_out$gen))) {
      next
    }
    fname_out <- define_fname_output(fname)
    write.table(df_out, file = fname_out, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
    print(paste0("Processed: `", fname_out, "`"))
  }
  NULL
}

prepare_gp_data <- function(fname_geno) {
  # fname_geno <- "maize_geno.csv"
  fname_pheno <- gsub("geno", "pheno", fname_geno)
  df_geno <- read.table(fname_geno, header = TRUE, sep = ",", check.names = FALSE)
  df_pheno <- read.table(fname_pheno, header = TRUE, sep = ",", check.names = FALSE)
  if (nrows(df_geno) != nrows(df_pheno)) {
    stop("Number of rows in genotype and phenotype files do not match.")
  }
  if (colnames(df_geno)[1] != "ID") {
    stop("First column in genotype file must be named 'ID'.")
  }
  if (colnames(df_pheno)[1] != "ID") {
    stop("First column in phenotype file must be named 'ID'.")
  }
  df_geno <- df_geno[order(df_geno$ID), ]
  df_pheno <- df_pheno[order(df_pheno$ID), ]
  if (!all(df_geno$ID == df_pheno$ID)) {
    stop("IDs in genotype and phenotype files do not match.")
  }
  for (j in seq_len(df_pheno)) {
    # j <- 2
    if (j == 1) {next}
    y_name <- names(df_pheno)[j]
    y <- df_pheno[, j]
    if (is.numeric(y) && (length(unique(y)) > 5) && (var(y, na.rm = TRUE) > 1e-7)) {
      df_out <- cbind(data.frame(y = y), df_geno[, -1, drop=FALSE])
      df_out = df_out[complete.cases(df_out), ]
      fname_out <- define_fname_output(fname_geno, y_name)
      write.table(df_out, file = fname_out, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
      print(paste0("Processed: `", fname_out, "`"))
    }
  }
  NULL
}