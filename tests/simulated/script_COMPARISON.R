fnames_INPUT = list.files(".", pattern="input")
fnames_LINEAR = list.files(".", pattern="output_.*-LINEAR")
fnames_MLP = list.files(".", pattern="output_.*-MLP")
for (fname_input in fnames_INPUT) {
    # fname_input = fnames_INPUT[1]
    id = gsub("input_simulated-", "", gsub(".tsv", "", fname_input))
    fname_linear = fnames_LINEAR[grep(id, fnames_LINEAR)]
    fname_mlp = fnames_MLP[grep(id, fnames_MLP)]
    # Load the effects from the best linear model
    df_linear = read.delim(fname_linear, T)
    if (length(grep("➵", df_linear$ids)) > 0) {
        df_linear = df_linear[grep("^entry", df_linear$ids), ]
        df_linear$ids = gsub("entry➵", "", df_linear$ids)
    }
    colnames(df_linear)[2] = "linear"
    # Load the marginal effects from mlp
    df_mlp = read.delim(fname_mlp, T)
    df_mlp = df_mlp[grep("^entry", df_mlp$ids), 1:2]
    df_mlp$ids = gsub("entry➵", "", df_mlp$ids)
    colnames(df_mlp)[2] = "mlp"
    # Merge
    df = merge(df_linear, df_mlp, by="ids")
    # Calculate the correlation and R2
    cortest = cor.test(df$linear, df$mlp)
    annot = if (cortest$p.value < 0.0001) {
        "***"
    } else if (cortest$p.value < 0.001) {
        "**"
    } else if (cortest$p.value < 0.01) {
        "*"
    } else {
        "ns"
    }
    R2 = mean(c(1 - (sum((df$linear - df$mlp)^2) / sum((df$linear - mean(df$linear))^2)), 1 - (sum((df$linear - df$mlp)^2) / sum((df$mlp - mean(df$mlp))^2))))
    # Plot
    fname_png = paste0("comparison-", id, ".png")
    linear_model_formula = gsub(paste0("output_simulated-", id, "-LINEAR_"), "", gsub(".tsv", "", fname_linear))
    png(fname_png)
    plot(df$linear, df$mlp, xlab=paste0("Linear Model Estimated Effects\n(", linear_model_formula, ")"), ylab="Multi-layer Perceptron\nMarginal Effects", main=id)
    grid()
    text(min(df$linear), max(df$mlp), label=paste0("\n\ncor=", round(100*cortest$estimate, 2), "%", annot, "\nR²=", round(R2, 2)), pos=c(4, 1))
    dev.off()
}
