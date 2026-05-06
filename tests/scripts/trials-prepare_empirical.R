fnames = list.files(path=".", pattern=".txt$")
fnames = fnames[!grepl(".covs.txt", fnames)]
fnames = fnames[!grepl(".uniformity.txt", fnames)]
for (fname in fnames) {
    # fname = fnames[261]
    # fname = "archbold.apple.txt"
    # fname = "acorsi.grayleafspot.txt"
    # fname = "alwan.lamb.txt"
    # fname = "aastveit.barley.height.txt"
    df = read.table(fname, header=TRUE, na.strings=c("", "NA", "NAN", "NaN", "na", "nan"))
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # print(fname)
    # print(str(df))
    # readline(prompt="Press [enter] to proceed")
    # }
    potential_explanatory_names = c(
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
        "year", "years",
        "loc", "locs",
        "harvest", "harvests",
        "season", "seasons",
        "plot", "plots",
        "rep", "reps",
        "row", "rows",
        "col", "cols",
        "blk", "blks",
        "genotype", "genotypes",
        "population", "populations",
        "variety", "varieties",
        "cultivar", "cultivars",
        "replication", "replications",
        "column", "columns",
        "block", "blocks",
        "pos", "position", "positions",
        "spacing", "spacings",
        "stock", "stocks",
        "trt", "trts",
        "treatment", "treatments"
    )
    potential_response_names = c(
        "yield",
        "grain",
        "straw",
        "height",
        "size",
        "lodging",
        "protein",
        "oil"
    )
    explanatory_names = c()
    response_names = c()
    for (j in 1:ncol(df)) {
        # j = 1
        id = names(df)[j]
        y = df[, j]
        n = length(y)
        if (id %in% potential_explanatory_names) {
            if (is.numeric(y)) {
                if ((length(unique(y)) > 5) && (var(y, na.rm=TRUE) > 1e-7)) {
                    response_names = c(response_names, id)
                } else {
                    explanatory_names = c(explanatory_names, id)
                    df[, j] = paste0(id, "➵", y)
                }
            } else {
                explanatory_names = c(explanatory_names, id)
            }
        } else {
            if (id %in% potential_response_names) {
                response_names = c(response_names, id)
            } else {
                if (is.character(y)) {
                    explanatory_names = c(explanatory_names, id)
                } else if ((length(unique(y)) < 5) | (var(y, na.rm=TRUE) < 1e-7)) {
                    explanatory_names = c(explanatory_names, id)
                    df[, j] = paste0(id, "➵", y)
                } else {
                    response_names = c(response_names, id)
                }
            }
        }
    }
    idx_explanatories = which(names(df) %in% explanatory_names)
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
    df_explanatories = df[, idx_explanatories, drop=FALSE]
    for (y_name in response_names) {
        # y_name = response_names[1]
        df_out = cbind(data.frame(y=df[, which(names(df) == y_name)]), df_explanatories)
        fname_out = gsub(".txt", paste0("-", y_name, ".tsv"), fname)
        write.table(df_out, file=fname_out, sep="\t", row.names=FALSE, col.names=TRUE, quote=FALSE)
        print(paste0("Processed: `", fname_out, "`"))
    }
}
