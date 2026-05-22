args = commandArgs(trailingOnly=TRUE)

if (args[1] == "-h" || args[1] == "--help") {
  cat("This script prepares empirical datasets for testing the modeling functions. It takes an input file, identifies the response and explanatory variables, and creates output files in a standardized format for analysis. The script can handle both trial data and genotype-phenotype data.\n")
  cat("Usage: Rscript empiricalprep.R ANALYSIS_TYPE FNAME_INPUT\n")
  cat("1. ANALYSIS_TYPE: 'trials' or 'gp'.\n")
  cat("2. FNAME_INPUT: path to the input file for the analysis:\n")
  cat("\t- For trials analysis (i.e. to extract the marginal effects of each genotype), this should be a tab-separated file with a header row and columns for year, site, treatment, entry, replication, and response variable.\n")
  cat("\t- For genomic prediction analysis (i.e. repeated k-fold cross-validation), this should be a tab-separated file with a header row and columns for the response variable followed by the features.\n")
  cat("Examples:\n")
  cat("DIR=${HOME}/Documents/mlp/tests\n")
  cat("cd ${DIR}/scripts\n")
  cat("mkdir tmp\n")
  cat("Rscript empiricalprep.R trials ${DIR}/datasets/agridat/australia.soybean.txt tmp\n")
  cat("Rscript empiricalprep.R gp ${DIR}/datasets/azodi_2019/sorghum_geno.csv tmp\n")
  quit(status = 0)
}

#' Extracts and prepares the parameters from the command line arguments, including validation of the analysis type and input file existence.
#' For trials data, it identifies potential response and explanatory variables based on column names and data characteristics, and creates output files for each response variable.
#' For gp (genomic prediction) data, it ensures that the genotype and phenotype files are properly formatted.
get_params <- function(args) {
  if (length(args) < 3) {
    stop("Three arguments are required: analysis_type, input file name, and output directory.")
  }
  params = list(
    analysis_type = args[1],
    fname_input = args[2],
    dirname_output = args[3]
  )
  if ((params$analysis_type != "trials") && (params$analysis_type != "gp")) {
    stop(paste0("Invalid analysis type: ", params$analysis_type, ". Please choose either 'trials' or 'gp'."))
  }
  if (!file.exists(params$fname_input)) {
    stop(paste0("Input file does not exist: ", params$fname_input))
  }
  if (!dir.exists(params$dirname_output)) {
    stop(paste0("Output directory does not exist: ", params$dirname_output))
  }
  if (params$analysis_type == "gp") {
    fname_pheno <- gsub("geno", "pheno", params$fname_input)
    if (!file.exists(fname_pheno)) {
      stop(paste0("Phenotype file does not exist: ", fname_pheno, ". We expect the genotype file to have a corresponding phenotype file with the same name but with 'geno' replaced by 'pheno'."))
    }
  }
  params
}

#' Defines the output file name based on the input file name and the response variable name. It replaces the extension with a standardized format and removes any "_geno" suffix for clarity.
define_fname_output <- function(fname, y_name, dirname_output) {
  fname_output <- gsub(".txt$", paste0("-", y_name, ".tsv"), basename(fname))
  fname_output <- gsub(".csv$", paste0("-", y_name, ".tsv"), fname_output)
  fname_output <- gsub("_geno", "", fname_output)
  file.path(dirname_output, fname_output)
}

#' Prepares trial data by reading the input file, identifying response and explanatory variables, and creating output files for each response variable. It handles various naming conventions and data characteristics to ensure that the output is suitable for analysis.
prepare_trial_data <- function(params) {
  if (grepl(".covs.txt", params$fname_input)) {
    stop("Covariate file found: ", params$fname_input)
  }
  if (grepl(".uniformity.txt", params$fname_input)) {
    stop("Uniformity file found: ", params$fname_input)
  }
  df <- read.table(params$fname_input, header = TRUE, na.strings = c("", "NA", "NAN", "NaN", "na", "nan"))
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
    fname_output <- define_fname_output(params$fname_input, y_name, params$dirname_output)
    write.table(df_out, file = fname_output, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
    print(paste0("Processed: `", fname_output, "`"))
  }
}

#' Prepares genomic prediction data by reading the genotype and phenotype files, ensuring they are properly formatted, and creating output files for each numeric response variable in the phenotype file. It checks for matching IDs and handles missing values appropriately.
prepare_gp_data <- function(params) {
  # params <- list(fname_input = "sorghum_geno.csv")
  fname_geno <- params$fname_input
  fname_pheno <- gsub("geno", "pheno", fname_geno)
  df_geno <- read.table(fname_geno, header = TRUE, sep = ",", check.names = FALSE)
  df_pheno <- read.table(fname_pheno, header = TRUE, sep = ",", check.names = FALSE)
  if (nrow(df_geno) != nrow(df_pheno)) {
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
  idx = which(complete.cases(df_geno))
  df_geno <- df_geno[idx, , drop=FALSE]
  df_pheno <- df_pheno[idx, , drop=FALSE]
  df_out <- data.frame( y = NA, df_geno[, -1, drop=FALSE])
  for (j in 2:ncol(df_pheno)) {
    # j <- 2
    y_name <- names(df_pheno)[j]
    y <- df_pheno[, j]
    if (is.numeric(y) && (length(unique(y)) > 5) && (var(y, na.rm = TRUE) > 1e-7)) {
      idx_y = which(!is.na(y))
      df_out$y <- y
      fname_output <- define_fname_output(fname_geno, y_name, params$dirname_output)
      write.table(df_out[idx_y, ], file = fname_output, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
      print(paste0("Processed: `", fname_output, "`"))
    }
  }
}

###########################################################
# Execute
###########################################################
# Testing: source("../../scripts/empiricalprep.R")
# args = c("trials", "australia.soybean.txt")
# args = c("gp", "sorghum_geno.csv")
params <- get_params(args)
if (params$analysis_type == "trials") {
  prepare_trial_data(params)
} else {
  prepare_gp_data(params)
}
