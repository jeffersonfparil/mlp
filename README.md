# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

# Quickstart

```shell
./mlp -h

Usage: mlp [OPTIONS]

Options:
  -f, --fname <FNAME>
          Input file name
  -d, --delim <DELIM>
          Delimiter for the input data file [default: "\t"]
  -t, --column-indices-of-targets <COLUMN_INDICES_OF_TARGETS>
          Vector of column indexes corresponding to the target values in the input data file [default: 0]
      --n-hidden-layers <N_HIDDEN_LAYERS>
          Number of hidden layers [default: 1]
      --n-hidden-nodes <N_HIDDEN_NODES>
          Number of nodes per hidden layer [default: 128]
      --dropout-rates <DROPOUT_RATES>
          Dropout rates per hidden layer [default: 0.0]
      --activation <ACTIVATION>
          Activation function (Choose from: "ReLU", "Sigmoid", "HyperbolicTangent") (Note: "LeakyReLU" under construction) [default: ReLU]
      --cost <COST>
          Cost function (Choose: "MSE", "MAE", "HL") [default: MSE]
      --optimiser <OPTIMISER>
          Optimiser (Choose: "Adam", "AdamMax", "GradientDescent") [default: Adam]
      --n-epochs <N_EPOCHS>
          Maximum number of training epochs [default: 10]
      --f-patient-epochs <F_PATIENT_EPOCHS>
          Fraction of the maximum number of epochs to wait before enabling the criteria for early stopping [default: 0.25]
      --n-batches <N_BATCHES>
          Number of training batches to split the input data into [default: 2]
      --learning-rate <LEARNING_RATE>
          Learning rate (η) [default: 0.001]
      --first-moment-decay <FIRST_MOMENT_DECAY>
          First moment decay (β₁) [default: 0.001]
      --second-moment-decay <SECOND_MOMENT_DECAY>
          Second moment decay (β₁) [default: 0.999]
      --epsilon <EPSILON>
          Small value used for numerical stability (ϵ; usually to avoid dividing by zero) [default: 0.00000001]
      --seed <SEED>
          Randomisation seed [default: 123]
  -o, --fname-network-output <FNAME_NETWORK_OUTPUT>
          Filename of the output model (Default: "output_network-{%Y%m%d%H%M%S}.json")
  -v, --verbose
          Verbose
      --hyperparameter-optimisation
          Hyperparameter optimisation
      --range-hidden-layers <RANGE_HIDDEN_LAYERS>
          Range of number of hidden layers for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 1,2,1]
      --range-hidden-layer-nodes <RANGE_HIDDEN_LAYER_NODES>
          Range of number of nodes per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 100,100,100]
      --range-dropout-rates <RANGE_DROPOUT_RATES>
          Range of dropout rates per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 0.0,0.0,0.01]
      --range-learning-rates <RANGE_LEARNING_RATES>
          Range of learning rates for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 1e-5,1e-5,1e-5]
      --range-n-epochs <RANGE_N_EPOCHS>
          Range of maximum number of training epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 10,10,10]
      --range-f-patient-epochs <RANGE_F_PATIENT_EPOCHS>
          Range of proportions of the maximum training epochs to start considering early stopping for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 0.5,1.0,0.5]
      --range-n-batches <RANGE_N_BATCHES>
          Range of number of batches to split the dataset for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 1,2,1]
      --selection-activations <SELECTION_ACTIVATIONS>
          Activation functions to test [default: ReLU]
      --selection-costs <SELECTION_COSTS>
          Cost functions to test [default: MSE]
      --selection-optimisers <SELECTION_OPTIMISERS>
          Optimisers to test [default: GradientDescent,Adam]
      --predict-only
          Predict using a fitted network (fitted MLP model)
  -m, --model <MODEL>
          File name of the MLP model in JSON format
  -M, --marginals-only
          Marginal effects estimation only
      --skip-marginals
          
      --marginals-order <MARGINALS_ORDER>
          Maximum number of interaction effects level, i.e. order 1 includes only the main effects, order 2 includes the main effects and pairwise interactions, and so on [default: 1]
      --n-interpolate-min-max <N_INTERPOLATE_MIN_MAX>
          Number of input values across the observed range per feature (or input node) to use in predictions i.e. number of values for interpolate between minimum and maximum values observed in each feature or input node [default: 10]
  -D, --deep-shap
          Use DeepSHAP instead of the perturbation method Note that the current implementation of DeepSHAP generates only main effects and no interaction effects. Do not enable this flag to use the default perturbation method if you require marginal interaction effects
      --deep-shap-reps <DEEP_SHAP_REPS>
          Number of replications for DeepSHAP main effects estimation Each replication samples feature values from their normally distributed values [default: 10]
  -s, --simulate-data-only
          Simulate data only
  -n, --simulation-n-observations <SIMULATION_N_OBSERVATIONS>
          Number of observations to simulate [default: 100]
  -p, --simulation-n-features-continuous <SIMULATION_N_FEATURES_CONTINUOUS>
          Number of continuous features to simulate [default: 10]
  -q, --simulation-n-features-categorical <SIMULATION_N_FEATURES_CATEGORICAL>
          Number of continuous features to simulate [default: 2,3,5]
  -k, --simulation-n-output-columns <SIMULATION_N_OUTPUT_COLUMNS>
          Number of simulated output column [default: 1]
  -l, --simulation-n-hidden-layers <SIMULATION_N_HIDDEN_LAYERS>
          Number of hidden layers to use to simulate the output data [default: 2]
      --simulation-weights-distribution <SIMULATION_WEIGHTS_DISTRIBUTION>
          Two-parameter distribution from which the simulated weights will be sample from Select from: "normal","lognormal","cauchy","weibull","gamma","beta" [default: normal]
      --simulation-weights-distribution-param-1 <SIMULATION_WEIGHTS_DISTRIBUTION_PARAM_1>
          First parameter of the distribution from which the weights will be sampled from [default: 0]
      --simulation-weights-distribution-param-2 <SIMULATION_WEIGHTS_DISTRIBUTION_PARAM_2>
          First parameter of the distribution from which the weights will be sampled from [default: 1]
  -h, --help
          Print help (see more with '--help')
  -V, --version
          Print version
```

# Development setup

1. Install [pixi](https://pixi.prefix.dev/):

```shell
wget -qO- https://pixi.sh/install.sh | sh
```

2. Setup the workspace:

```shell
git clone https://github.com/jeffersonfparil/mlp.git
cd mlp
pixi init
```

3. Add cargo, cuda-nvrtc:

```shell
cd mlp
pixi shell
pixi add rust
pixi add cuda-nvrtc==12.8.93
which cargo
ls -lhtr ${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib/libnvrtc*
```

# Unit testing

```shell
cd mlp
pixi shell
# export LD_LIBRARY_PATH=${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib
time cargo test -- --show-output
```

# More testing

```shell
cd mlp
pixi shell
# export LD_LIBRARY_PATH=${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib
time cargo run -- -h
time cargo run -- -s -n1000 -p10 -v


# TESTING CROSS-VALIDATION FOR N << P DATASETS
time cargo run -- -s -n500 -p12000 -q0 -v # 13 minutes on gpu001: Intel(R) Xeon(R) Gold 5418Y 24 cores with 1 NVIDIA H100 NVL (93.584Gi)
INPUT=$(ls -t1 | grep "input.*.tsv" | head -n1)
N=$(cat $INPUT | wc -l)
V=$(printf %.0f $(echo  "scale=0; $N * 0.1" | bc))
T=$(echo  "$N - $V" | bc)

echo $N
echo $V
echo $T

head -n$T $INPUT > training_data.tsv
head -n1 $INPUT > validation_data.tsv
tail -n$V $INPUT >> validation_data.tsv
time cargo run -- -f training_data.tsv -o output.json -v --n-batches=1 --n-epochs=1000 --f-patient-epochs=0.5 --skip-marginals
time cargo run -- -f validation_data.tsv -m output.json -v --predict-only
PREDICTED=$(ls -t1 | grep "output.*-predictions.tsv" | tail -n1)
echo $PREDICTED

cut -f1 validation_data.tsv > true.tmp
cut -f1 $PREDICTED > pred.tmp
paste -d'\t' true.tmp pred.tmp > true_vs_pred.tsv
head true_vs_pred.tsv

# R --> ...
# df = read.table("true_vs_pred.tsv", T)
# cor(df)
# plot(df[, 1], df[, 2])
# dev.off()



INPUT=$(ls -t1 | grep "input.*.tsv" | head -n1)
head $INPUT | cut -f1-10
head -n1 $INPUT | awk '{print NF}'
time cargo run -- -f $INPUT -v --n-batches=1 --n-epochs=1000 --skip-marginals
MODEL=$(ls -t1 | grep "output.*.json" | head -n1)
head $MODEL | cut -f1-10
tail $MODEL | cut -f1-10
time cargo run -- -f $INPUT -v -m $MODEL --predict-only
PREDICTED=$(ls -t1 | grep "output.*-predictions.tsv" | tail -n1)
head $PREDICTED | cut -f1-10
time cargo run -- -f $INPUT -v -m $MODEL --marginals-only --marginals-order=1
MARGINALS=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
mv $MARGINALS marginal_main.tsv
time cargo run -- -f $INPUT -v -m $MODEL --marginals-only --marginals-order=2
MARGINALS=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
mv $MARGINALS marginal_2nd.tsv
time cargo run -- -f $INPUT -v -m $MODEL --marginals-only --marginals-order=3
MARGINALS=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
mv $MARGINALS marginal_3rd.tsv
time cargo run -- -f $INPUT -v -m $MODEL --marginals-only --deep-shap --deep-shap-reps=100
MARGINALS=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
mv $MARGINALS deep_shap.tsv
head marginal_main.tsv
head marginal_2nd.tsv
head marginal_3rd.tsv
head deep_shap.tsv
```

# Compile for release

```shell
cd mlp
cargo build --release
./target/release/mlp -h
```

# Example Fits

Using 2 hidden layers, 128 nodes per hidden layer, ReLU activation, Adam optimiser, 0.001 learning rate and, 25% patient epochs:

## 10 Epochs

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E10-FPE0.25-B1-LR0.001-T20260408052818.svg)

## 20 Epochs

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E20-FPE0.25-B1-LR0.001-T20260408052908.svg)

## 50 Epochs

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E50-FPE0.25-B1-LR0.001-T20260408053111.svg)

## 100 Epochs

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E100-FPE0.25-B1-LR0.001-T20260408053517.svg)

# Special characters

- Used in progress bars: `█`
- Used as delimiters between non-numeric or categorical variable names and their levels: `➵`
- Used as delimiters in marginals' combinations: `▓`

# Field trial analysis

What we want to show here is that MLP estimates genotype effects similar to those of linear models while also getting better model fit.

Evaluation of the performance of MLP model on yield trial data:

- simulated multi-environment field trials datasets (multiple years, sites, treatments, entries, and replications in RCBD)
- empirical data from [agridat](https://kwstat.github.io/agridat/)
- comparison against linear models in R: selecting the best model among `lm`, `lmer`, and `asreml` (if available) models

## Tests on simulated data

We simulated:
- 21,000 observations of
- 1 response variable across 
- 7 years
- 20 sites
- 2 treatments
- 25 entries
- 3 replications (blocks)
- in a multi-environment trial
- in randomised complete block design (RCBD) per environment
- using a multi-layer perceptron with:
    + 1, 2, 3, 4, and 5 hidden layers whose effects are:
    + normally (μ=0, σ=1) and gamma (α=2, θ=2) distributed.

<details>

### Simulate data

```shell
cd mlp/
cd tests/trials/simulated
MLP=../../../target/release/mlp
N_YEARS_SMALL=2
N_YEARS_LARGE=7
N_SITES_SMALL=2
N_SITES_LARGE=20
N_TREATMENTS_SMALL=2
N_TREATMENTS_LARGE=5
N_ENTRIES_SMALL=25
N_ENTRIES_LARGE=100
N_REPLICATIONS=3
for HIDDEN_LAYERS in $(seq 1 5)
do
    # HIDDEN_LAYERS=1
    F_SMALL=input_simulated-SMALL-${HIDDEN_LAYERS}HL.tsv
    F_LARGE=input_simulated-LARGE-${HIDDEN_LAYERS}HL.tsv
    echo "######################"
    echo "$F_SMALL and $F_LARGE"
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $(echo "$N_YEARS_SMALL*$N_SITES_SMALL*$N_TREATMENTS_SMALL*$N_ENTRIES_SMALL*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS_SMALL,$N_SITES_SMALL,$N_TREATMENTS_SMALL,$N_ENTRIES_SMALL,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed ${HIDDEN_LAYERS}
    F=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    sed 's/target_0/y/g' $F | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > tmp
    mv tmp $F
    mv $F $F_SMALL
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $(echo "$N_YEARS_LARGE*$N_SITES_LARGE*$N_TREATMENTS_SMALL*$N_ENTRIES_SMALL*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS_LARGE,$N_SITES_LARGE,$N_TREATMENTS_SMALL,$N_ENTRIES_SMALL,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed ${HIDDEN_LAYERS}
    F1=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    sed 's/target_0/y/g' $F1 | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > tmp
    mv tmp $F1
    mv $F1 $F_LARGE
done
```

### Analysis using R

Run `tests/trials/simulated/script_LINEAR.R`:

```shell
cd mlp
cd tests/trials/simulated
module load ASReml-R # if ASReml-R is available
time Rscript script_LINEAR.R
```

### Analysis using mlp

Run `tests/trials/simulated/script_MLP.sh`:

```shell
cd mlp
cd tests/trials/simulated
time sh script_MLP.sh
```

### Comparison between linear mixed model and mlp

#### Run the script:

```shell
cd mlp
cd tests/trials/simulated
time Rscript script_COMPARISON.R
```

#### See details below (`tests/trials/simulated/script_COMPARISON.R`):

```R
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
```

</details>

#### GAMMA-1HL

![](./tests/trials/simulated/comparison-GAMMA-1HL.png)

#### GAMMA-2HL

![](./tests/trials/simulated/comparison-GAMMA-2HL.png)

#### GAMMA-3HL

![](./tests/trials/simulated/comparison-GAMMA-3HL.png)

#### GAMMA-4HL

![](./tests/trials/simulated/comparison-GAMMA-4HL.png)

#### GAMMA-5HL

![](./tests/trials/simulated/comparison-GAMMA-5HL.png)

#### NORMAL-1HL

![](./tests/trials/simulated/comparison-NORMAL-1HL.png)

#### NORMAL-2HL

![](./tests/trials/simulated/comparison-NORMAL-2HL.png)

#### NORMAL-3HL

![](./tests/trials/simulated/comparison-NORMAL-3HL.png)

#### NORMAL-4HL

![](./tests/trials/simulated/comparison-NORMAL-4HL.png)

#### NORMAL-5HL

![](./tests/trials/simulated/comparison-NORMAL-5HL.png)



## Tests on empirical data

Using agridat data... details: how many? types?

<details>

### Prepare test data

- Explanatory variables can be numeric or categorical which means that explanatory variables which are written as numeric but are meant to be categorical need to be converted into strings, e.g. convert `rep=[1, 1, 1, 2, 2, 2, 3, 3]` into `rep=["rep➵1", "rep➵1", "rep➵1", "rep➵2", "rep➵2", "rep➵2", "rep➵3", "rep➵3"]`.
- We exclude covariance and uniformity datasets.
- We only include datasets with the genotype (`gen`) variable because we are testing `mlp`'s applicability for breeding, i.e. we want to rank the genotypes for selection.
- We then generate one file with a single response variable named `y` in the first column followed by the explanatory variables.

#### Download agridat data:

```shell
cd mlp/
cd tests/trials
mkdir agridat/
cd agridat/
curl -L https://codeload.github.com/kwstat/agridat/tar.gz/main | tar -xz --strip=2 agridat-main/data
```

#### Prepare the data, i.e. make sure categorical variables are interpretted as strings

```R
setwd("tests/trials/agridat")
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
```

### R

Fit linear models in R using `lm`, `lmer` and `asreml` (if available)

#### TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
#### TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
#### TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO


#### Generic model selection

##### Run:

```shell
cd tests/trials/agridat
time Rscript ...
```

##### Details (`...`):

```R
library("stringr")
library("lme4")
if (nzchar(system.file(package = "asreml"))) {
    library("asreml") # requires ```shell module load ASReml-R ```
}

# length(list.files(path=".", pattern=".tsv$"))

process_features = function(df) {
    # fname = list.files(path=".", pattern=".tsv$")[19]; df = read.table(fname, sep="\t", header=TRUE, na.strings=c("", "NA", "NAN", "NaN", "na", "nan"))
    # Assuming only the first column is the numeric response variable and the rest are non-numeric explanatory variables
    ids_features = colnames(df)[2:ncol(df)]
    for (j in 2:ncol(df)) {
        df[, j] = as.factor(df[, j])
    }
    if (length(ids_features) > 2) {
        idx = which((names(df) != "y") & (names(df) != "gen"))
        df$dummy_env = apply(df[, idx, drop=FALSE], MARGIN=1, FUN=function(x){paste(x, collapse="|")})
        ids_features = c(ids_features, "dummy_env")
    }
    # str(df)
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
        # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
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
        # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
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
        # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
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
        # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
        NA
    }
}

fit_extract_effects = function(df) {

    # fname = list.files(path=".", pattern=".tsv$")[22]; df = process_features(read.table(fname, sep="\t", header=TRUE, na.strings=c("", "NA", "NAN", "NaN", "na", "nan")))[["df"]]
    str(df)

    x_names = colnames(df)[2:ncol(df)]
    x_names_except_gen_and_dummy_env = x_names[(x_names != "gen") & (x_names != "dummy_env")]


    # TODO: define a bunch of sensible models

    lm_model_strings = c(
        "lm(y ~ ., data=df)",
        "lm(y ~ gen, data=df)",
        "lm(y ~ dummy_env + gen, data=df)",
        "lm(y ~ dummy_env*gen, data=df)",
        paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + gen, data=df)"),
        paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + dummy_env + gen, data=df)"),
        paste0("lm(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + dummy_env*gen, data=df)"),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("lm(y ~ ", x, " + gen, data=df)")})),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("lm(y ~ ", x, "*gen, data=df)")}))
    )
    m = length(x_names_except_gen_and_dummy_env)
    if (m > 1) {
        for (i in 1:(m-1)) {
            x1 = x_names_except_gen_and_dummy_env[i]
            for (j in (i+1):m) {
                x2 = x_names_except_gen_and_dummy_env[j]
                lm_model_strings = c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen, data=df)"))
                lm_model_strings = c(lm_model_strings, paste0("lm(y ~ ", x1, " + ", x2, " + gen + ", x1, ":gen, data=df)"))
                lm_model_strings = c(lm_model_strings, paste0("lm(y ~ ", x1, "*", x2, " + gen, data=df)"))
            }
        }
    }
    lm_model_strings

    lmer_model_strings = c(
        "lmer(y ~ (1|gen), data=df)",
        "lmer(y ~ dummy_env + (1|gen), data=df)",
        "lmer(y ~ dummy_env + (1|gen:dummy_env), data=df)",
        "lmer(y ~ (1|gen:dummy_env), data=df)",
        "lmer(y ~ (1|dummy_env) + (1|gen:dummy_env), data=df)",
        paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + (1|gen), data=df)"),
        paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + dummy_env + (1|gen), data=df)"),
        paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + (1|gen:dummy_env), data=df)"),
        paste0("lmer(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), " + dummy_env + (1|gen:dummy_env), data=df)"),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("lmer(y ~ ", x, " + (1|gen), data=df)")})),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("lmer(y ~ ", x, " + (1|gen:", x, "), data=df)")})),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("lmer(y ~ (1|gen:", x, "), data=df)")}))
    )
    m = length(x_names_except_gen_and_dummy_env)
    if (m > 1) {
        for (i in 1:(m-1)) {
            x1 = x_names_except_gen_and_dummy_env[i]
            for (j in (i+1):m) {
                x2 = x_names_except_gen_and_dummy_env[j]
                lmer_model_strings = c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen), data=df)"))
                lmer_model_strings = c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen:", x1, "), data=df)"))
                lmer_model_strings = c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen:", x2, "), data=df)"))
                lmer_model_strings = c(lmer_model_strings, paste0("lmer(y ~ ", x1, " + ", x2, " + (1|gen:", x1, ") + (1|gen:", x2, "), data=df)"))
            }
        }
    }
    lmer_model_strings


    asreml_model_strings = c(
        "asreml(y ~ 1, random = ~ gen, data=df)",
        "asreml(y ~ dummy_env, random = ~ gen, data=df)",
        "asreml(y ~ dummy_env, random = ~ gen:dummy_env, data=df)",
        "asreml(y ~ 1, random = ~ gen:dummy_env, data=df)",
        "asreml(y ~ 1, random = ~ dummy_env + gen:dummy_env, data=df)",
        paste0("asreml(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), ", random = ~ gen, data=df)"),
        paste0("asreml(y ~ ", paste(x_names_except_gen_and_dummy_env, collapse=' + ' ), ", random = ~ gen + fa(dummy_env):gen, data=df)"),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("asreml(y ~ ", x, ", random = ~ gen, data=df)")})),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("asreml(y ~ ", x, ", random = ~ ", x, ":gen, data=df)")})),
        unlist(lapply(x_names_except_gen_and_dummy_env, FUN=function(x){paste0("asreml(y ~ ", x, ", random = ~ fa(", x, "):gen, data=df)")}))
    )
    m = length(x_names_except_gen_and_dummy_env)
    if (m > 1) {
        for (i in 1:(m-1)) {
            x1 = x_names_except_gen_and_dummy_env[i]
            for (j in (i+1):m) {
                x2 = x_names_except_gen_and_dummy_env[j]
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ ", x1, ":gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ ", x2, ":gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ ", x1, ":gen + ", x2, ":gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ fa(", x1, "):gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ fa(", x2, "):gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ fa(", x1, "):gen + ", x2, ":gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ ", x1, ":gen + fa(", x2, "):gen, data=df)"))
                asreml_model_strings = c(asreml_model_strings, paste0("asreml(y ~ ", x1, " + ", x2, ", random = ~ fa(", x1, "):gen + fa(", x2, "):gen, data=df)"))

            }
        }
    }
    asreml_model_strings

    model_strings = if (nzchar(system.file(package = "asreml"))) {
        c(lm_model_strings, lmer_model_strings, asreml_model_strings)
    } else {
        c(lm_model_strings, lmer_model_strings)
    }
    # Fit these models
    model_candidates = list()
    for (i in 1:length(model_strings)) {
        # i = 1
        # i = 40
        # i = length(model_strings)
        mod_string = model_strings[i]
        mod_label = unlist(strsplit(mod_string, "\\("))[1]
        print(paste0("Fitting ", mod_label, "_", i, ": `", mod_string, "`"))
        mod = tryCatch(
            {
                setTimeLimit(30)
                eval(parse(text=mod_string))
            },
            error = function(e) {
                print("Unable to fit: skipped!")
                return(NA)
            }
        )
        if ((length(mod) == 1) && is.na(mod)) {
            model_candidates[[paste0(mod_label, "_", i)]] = NA
        } else {
            if (class(mod) == "lmerMod") {
                if (mod@optinfo$conv$opt != 0) {
                    # Failed to converge
                    model_candidates[[paste0(mod_label, "_", i)]] = NA
                } else {
                    model_candidates[[paste0(mod_label, "_", i)]] = mod
                }
            } else if (class(mod) == "asreml") {
                if (mod$converge == FALSE) {
                    model_candidates[[paste0(mod_label, "_", i)]] = NA
                } else {
                    model_candidates[[paste0(mod_label, "_", i)]] = mod
                }
            } else {
                model_candidates[[paste0(mod_label, "_", i)]] = mod
            }
        }
    }
    df_stats = data.frame(
        model = names(model_candidates),
        formula = model_strings,
        AIC = sapply(model_candidates, AIC_lm_lmer_asreml),
        BIC = sapply(model_candidates, BIC_lm_lmer_asreml),
        logLik = sapply(model_candidates, logLik_lm_lmer_asreml)
    )
    idx_filter = which(!is.na(df_stats$AIC) & is.finite(df_stats$AIC))
    df_stats = df_stats[idx_filter, ]
    model_candidates_ORIG = model_candidates
    model_candidates = model_candidates[idx_filter]
    print(df_stats)
    z_AIC = scale(df_stats$AIC, scale=T, center=T)
    z_BIC = scale(df_stats$BIC, scale=T, center=T)
    z_logLik = -scale(df_stats$logLik, scale=T, center=T)
    df_stats$z_sum = 0.2*z_AIC + 0.6*z_BIC + 0.2*z_logLik
    print(df_stats)
    # Select the best model based on z_sum
    # best_model_idx = which.min(df_stats$BIC)
    # best_model_idx = which.min(df_stats$z_sum)
    best_model_idx = tail(which(df_stats$z_sum == min(df_stats$z_sum)), 1)
    best_model = model_candidates[[best_model_idx]]
    best_model_formula = df_stats$formula[best_model_idx]
    print(paste("Best model selected:", best_model_formula))


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
## TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
## TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
## TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    # Plot gen effects (random effects for gen)
    # best_model = model_candidates[[1]]
    df_effects = if (class(best_model) == "lm") {
        # best_model = model_candidates[[1]]
        effects = coef(best_model)
        ids = names(effects)
        intercept = effects[ids == "(Intercept)"]
        gen_effects = c(intercept, intercept + effects[grepl("gen", ids)])
        gen_names = c(as.character(levels(df$gen)[1]), ids[grepl("gen", ids)])
        gen_names = gsub("gen", "", gen_names)
        df_effects = data.frame(ids=gen_names, effects=gen_effects)
        rownames(df_effects) = NULL
        df_effects
        # barplot(gen_effects, names.arg=gen_names, main = "Estimated Entry Effects (fixed effects model)", xlab = "Entry", ylab = "Coefficients")
    } else if (class(best_model) == "lmerMod") {
        # best_model = model_candidates[[3]]
        gen_effects <- ranef(best_model)$gen
        df_effects = data.frame(ids=rownames(gen_effects), effects=gen_effects[,1])
        rownames(df_effects) = NULL
        df_effects
        # barplot(gen_effects[,1], names.arg = rownames(gen_effects), main = "Estimated Entry Effects (mixed model)", xlab = "Entry", ylab = "Random Effect")
    } else if (class(best_model) == "asreml") {
        # best_model = model_candidates[[13]]
        df_effects = data.frame(
            ids = rownames(coef(best_model)$random),
            effects = as.vector(coef(best_model)$random)
        ); row.names(df_effects) = NULL
        # str(df_effects)
        df_sub = df_effects[grepl("gen", df_effects$ids) & !grepl(":", df_effects$ids), ]
        df_sub$ids = gsub("gen_", "", df_sub$ids)
        df_effects
        # barplot(df_sub$effects, names.arg = df_sub$ids, main = "Estimated Entry Effects (asreml model)", xlab = "Entry", ylab = "Random Effect")
    } else {
        data.frame()
        # plot(0, 0)
        # print("Unknown model class. We expect 'lm', 'lmerMod' or 'asreml'.")
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
# fnames = list.files(path=".", pattern="input_simulated")
# output = list()
# for (fname_input in fnames) {
#     # fname_input = fnames[1]
#     input_list = process_features(df=read.delim(fname_input, T))
#     df = input_list$df
#     ids_features = input_list$ids_features
#     attach(df)
#     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#     print(fname_input)
#     out = fit_extract_effects(df)
#     fname_output = paste0(
#         gsub("^input", "output", gsub(".tsv", "", fname_input)),
#         "-LINEAR_",
#         gsub(" ", "", out$formula), 
#         ".tsv"
#     )
#     write.table(out$df_effects, file=fname_output, row.names=FALSE, col.names=TRUE, sep="\t")
#     output[[fname_input]] = out
#     detach(df)
# }

setwd("tests/trials/agridat")
fnames = list.files(path=".", pattern=".tsv$")
stopifnot(length(fnames) == 319)

for (fname in fnames) {
    # fname = fnames[19]
    df = read.table(fname, sep="\t", header=TRUE, na.strings=c("", "NA", "NAN", "NaN", "na", "nan"))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(fname)
    print(str(df))



}


```

#### Dataset-specific analyses

##### ...
##### ...
##### ...
##### ...
##### ...

### mlp

```shell
cd mlp/
cd tests/trials/agridat/
MLP=../../../target/release/mlp

for FILE in $(find . -name "*.txt")
do
    # FILE=$(find . -name "*.txt" | sort | head -n1 | tail -n1)
    # FILE=archbold.apple.txt
    # FILE=acorsi.grayleafspot.txt
    # FILE=alwan.lamb.txt
    echo $FILE
    DIR_OUTPUT=OUTPUT-$(basename ${FILE%.txt*})
    echo $DIR_OUTPUT
    mkdir $DIR_OUTPUT
    FILE_INPUTS=$(julia +1.12 --project=. scripts/prep_agridat.jl $FILE 0.1)
    echo $FILE_INPUTS
    # head $FILE_INPUTS
    if [ "$FILE_INPUTS" == "" ]
    then
        echo "Skipping empty explanatory/response variables"
    else
        for FILE_INPUT in $FILE_INPUTS
        do
            # FILE_INPUT=$(echo $FILE_INPUTS | cut -d' ' -f1)
            FILE_OUTPUT=${FILE_INPUT%.tsv*}.json
            echo $FILE_OUTPUT
            time $MLP \
                -f $FILE_INPUT \
                -o $FILE_OUTPUT \
                -t 0 \
                -v \
                --n-batches 1 \
                --n-hidden-layers 3 \
                --n-epochs 100 \
                --marginals-order 2 \
            > out.tmp
            FNAME_LOSS=$(grep "Find the loss curve saved as:" out.tmp | cut -d ':' -f2 | cut -d ' ' -f2)
            FNAME_SCAT=$(grep "Find the observed vs predicted scatterplot saved as:" out.tmp | cut -d ':' -f2 | cut -d ' ' -f2)
            FNAME_BARM=$(grep "Find the marginal effects" out.tmp | cut -d ':' -f2 | cut -d ' ' -f2)
            FNAME_MODEL=$(grep "Please find the output model (network) in json format:" out.tmp | cut -d ':' -f2 | cut -d ' ' -f2)
            FNAME_MARGINALS=$(grep "Please find the estimated marginal effects in tab-delimited format:" out.tmp | cut -d ':' -f2 | cut -d ' ' -f2)
            mv $FNAME_MODEL ${DIR_OUTPUT}/$(basename $FNAME_MODEL)
            mv $FNAME_MARGINALS ${DIR_OUTPUT}/$(basename $FNAME_MARGINALS)
            mv $FNAME_LOSS ${DIR_OUTPUT}/${FILE_INPUT%.tsv*}-$(basename $FNAME_LOSS)
            mv $FNAME_SCAT ${DIR_OUTPUT}/${FILE_INPUT%.tsv*}-$(basename $FNAME_SCAT)
            mv $FNAME_BARM ${DIR_OUTPUT}/${FILE_INPUT%.tsv*}-$(basename $FNAME_BARM)
            rm out.tmp
            rm $FILE_INPUT
        done
    fi
done
```

</details>


# Genomic prediction

## Tests on simulated data

### Simulate data

```shell
cd mlp/
mkdir tests/gp/simulated
cd tests/gp/simulated
MLP=../../../target/release/mlp
N=700
P=42000
for HIDDEN_LAYERS in $(seq 1 3) # cannot have more hidden layers because of GPU memory limitations (H100s and V100s)
do
    # HIDDEN_LAYERS=3
    F_CONTINUOUS=input_simulated-CONTINUOUS-${HIDDEN_LAYERS}HL.tsv
    F_BINARY=input_simulated-BINARY-${HIDDEN_LAYERS}HL.tsv
    echo "######################"
    echo "$F_CONTINUOUS and $F_BINARY"
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $N \
        --simulation-n-features-continuous $P \
        --simulation-n-features-categorical 0 \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed ${HIDDEN_LAYERS} \
        --verbose
    F0=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    mv $F0 $F_CONTINUOUS
    # Convert into binary genotype data
    head -n1 $F_CONTINUOUS > $F_BINARY
    tail -n+2 $F_CONTINUOUS | 
      awk '{
          FS="\t"; OFS="\t"; 
          for (i=2; i<=NF; i++) {
              $i = sprintf("%.0f", $i)
          }
      }{print}' - >> $F_BINARY
done
```

### BGLR??

TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 

Starting to draft Bayes A, B and C and also simple OLS down under MLP...

### MLP

```shell
cd tests/gp/simulated
MLP=../../../target/release/mlp
N_EPOCHS=1000
F_PATIENT_EPOCHS=0.50
F_VALIDATION=0.0
N_BATCHES=1
N_HIDDEN_LAYERS=1
N_HIDDEN_NODES=256

N_REPS=3
N_FOLDS=10



for INPUT in $(ls input_simulated-*-*.tsv)
do
    # INPUT=$(ls input_simulated-*-*.tsv | head -n4 | tail -n1)
    OUTPUT=$(echo $INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
    N=$(echo $(cut -f1 $INPUT | wc -l) - 1 | bc)
    M=$(echo "scale=0; $N / $N_FOLDS" | bc)
    echo "$INPUT -->  $OUTPUT (N=$N; M=$M)"


    for REP in $(seq 0 $N_REPS)
    do
        IDX_SHUFFLED=($(shuf --random-source=<(yes 42) -e $(seq 2 $(echo $N + 1 | bc))))
        echo ${IDX_SHUFFLED[@]}
        for FOLD in $(seq 0 $N_FOLDS)
        do
            # FOLD=1
            IDX_INI=$(echo "(($FOLD - 1) * $M) + 1" | bc)
            IDX_FIN=$(echo "$FOLD * $M" | bc)
            IDX_TRAINING=()
            IDX_VALIDATION=()
            for i in $(seq 0 $N)
            do
                if [[ ($i -ge $IDX_INI) && ($i -le $IDX_FIN) ]]
                then
                    # echo "$i; ${IDX_SHUFFLED[i]}"
                    IDX_VALIDATION+=("${IDX_SHUFFLED[i]}")
                else
                    IDX_TRAINING+=("${IDX_SHUFFLED[i]}")
                fi
            done
            echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
            echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
            echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"

            TRAINING_IDX=${IDX_TRAINING[@]}
            VALIDATION_IDX=${IDX_VALIDATION[@]}
            head -n1 $INPUT > TRAINING_SET.tmp
            head -n1 $INPUT > VALIDATION_SET.tmp
            awk -v idx="$TRAINING_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> TRAINING_SET.tmp
            awk -v idx="$VALIDATION_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> VALIDATION_SET.tmp
            time ${MLP} \
                -f TRAINING_SET.tmp \
                -o OUTPUT.tmp.json \
                -v \
                --n-epochs=${N_EPOCHS} \
                --f-patient-epochs=${F_PATIENT_EPOCHS} \
                --f-validation=${F_VALIDATION} \
                --n-batches=${N_BATCHES} \
                --n-hidden-layers=${N_HIDDEN_LAYERS} \
                --n-hidden-nodes=${N_HIDDEN_NODES} \
                --skip-marginals
            time ${MLP} \
                -f VALIDATION_SET.tmp \
                -m OUTPUT.tmp.json \
                -v \
                --predict-only


            cut -f1 VALIDATION_SET.tmp > true.tmp
            cut -f1 OUTPUT.tmp-predictions.tsv > pred.tmp
            paste -d'\t' true.tmp pred.tmp > true_vs_pred.tsv
            head true_vs_pred.tsv
            R
            df_mlp = read.table("true_vs_pred.tsv", T)
            str(df_mlp)
            cor(df_mlp)

            df_training = read.table("TRAINING_SET.tmp", T)
            df_validation = read.table("VALIDATION_SET.tmp", T)
            X = as.matrix(df_training[, 2:ncol(df_training)])
            y = df_training[, 1]
            
            # OLS
            b_hat = t(X) %*% solve(X %*% t(X)) %*% y
            y_hat = as.matrix(df_validation[, 2:ncol(df_validation)]) %*% b_hat
            cor(df_validation[, 1], y_hat)

            # Bayesian models
            library(BGLR)
            df = rbind(df_training, df_validation)
            X = as.matrix(df[, 2:ncol(df)])
            y = df[, 1]
            yNA = y
            idx_validation = nrow(df_training):nrow(df)
            yNA[idx_validation] = NA
            nIter=6000; burnIn=1000

            # Bayes A
            time = Sys.time()
            fmBA=BGLR(y=yNA,ETA=list( list(X=X,model='BayesA')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
            print(Sys.time() - t)
            yHat=fmBA$yHat[idx_validation]
            cor(yHat, y[idx_validation])

            # Bayes B
            time = Sys.time()
            fmBB=BGLR(y=yNA,ETA=list( list(X=X,model='BayesB')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
            print(Sys.time() - t)
            yHat=fmBB$yHat[idx_validation]
            cor(yHat, y[idx_validation])

            # Bayes C
            time = Sys.time()
            fmBC=BGLR(y=yNA,ETA=list( list(X=X,model='BayesC')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
            print(Sys.time() - t)
            yHat=fmBC$yHat[idx_validation]
            cor(yHat, y[idx_validation])


            



        done
    done


    
done




```



## Tests on empirical data

### BGLR??

TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 
TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO 

### MLP

```shell
cd tests/gp/azodi_2019
MLP=../../../target/release/mlp
N_EPOCHS=1000
F_PATIENT_EPOCHS=0.50
F_VALIDATION=0.0
N_BATCHES=1
N_HIDDEN_LAYERS=1
N_HIDDEN_NODES=256

N_REPS=3
N_FOLDS=10



for SPECIES in ("maize" "rice" "sorghum" "soy" "spruce" "switchgrass")
do
    # SPECIES=maize

    cut -d, -f1 ${SPECIES}_pheno.csv > ids_pheno.tmp
    cut -d, -f1 ${SPECIES}_geno.csv > ids_geno.tmp
    if [[ $(diff ids_pheno.tmp ids_geno.tmp | wc -l) -ne 0 ]]
    then
        echo "ERROR"
    fi

    T=$(head -n1 ${SPECIES}_pheno.csv | awk -F, '{print NF}')
    for j in $(seq 2 $T)
    do
        # j=2
        cut -d, -f$j ${SPECIES}_pheno.csv > y.tmp
        cut -d, -f2- ${SPECIES}_geno.csv > X.tmp
        paste -d, y.tmp X.tmp | sed -z 's/,/\t/g' > ${SPECIES}.tsv
        # wc -l ${SPECIES}.tsv
        # head -n1 ${SPECIES}.tsv | awk '{print NF}'
        # bat --wrap never -l tsv ${SPECIES}.tsv
        INPUT=${SPECIES}.tsv
        OUTPUT=$(echo $INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
        N=$(echo $(cut -f1 $INPUT | wc -l) - 1 | bc)
        M=$(echo "scale=0; $N / $N_FOLDS" | bc)
        echo "$INPUT -->  $OUTPUT (N=$N; M=$M)"
        for REP in $(seq 0 $N_REPS)
        do
            IDX_SHUFFLED=($(shuf --random-source=<(yes 42) -e $(seq 2 $(echo $N + 1 | bc))))
            echo ${IDX_SHUFFLED[@]}
            for FOLD in $(seq 0 $N_FOLDS)
            do
                # FOLD=1
                IDX_INI=$(echo "(($FOLD - 1) * $M) + 1" | bc)
                IDX_FIN=$(echo "$FOLD * $M" | bc)
                IDX_TRAINING=()
                IDX_VALIDATION=()
                for i in $(seq 0 $N)
                do
                    if [[ ($i -ge $IDX_INI) && ($i -le $IDX_FIN) ]]
                    then
                        # echo "$i; ${IDX_SHUFFLED[i]}"
                        IDX_VALIDATION+=("${IDX_SHUFFLED[i]}")
                    else
                        IDX_TRAINING+=("${IDX_SHUFFLED[i]}")
                    fi
                done
                echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
                echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"

                TRAINING_IDX=${IDX_TRAINING[@]}
                VALIDATION_IDX=${IDX_VALIDATION[@]}
                head -n1 $INPUT > TRAINING_SET.tmp
                head -n1 $INPUT > VALIDATION_SET.tmp
                awk -v idx="$TRAINING_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> TRAINING_SET.tmp
                awk -v idx="$VALIDATION_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> VALIDATION_SET.tmp
                wc -l TRAINING_SET.tmp
                wc -l VALIDATION_SET.tmp
                time ${MLP} \
                    -f TRAINING_SET.tmp \
                    -o OUTPUT.tmp.json \
                    -v \
                    --n-epochs=${N_EPOCHS} \
                    --f-patient-epochs=${F_PATIENT_EPOCHS} \
                    --f-validation=${F_VALIDATION} \
                    --n-batches=${N_BATCHES} \
                    --n-hidden-layers=${N_HIDDEN_LAYERS} \
                    --n-hidden-nodes=${N_HIDDEN_NODES} \
                    --skip-marginals
                time ${MLP} \
                    -f VALIDATION_SET.tmp \
                    -m OUTPUT.tmp.json \
                    -v \
                    --predict-only


                cut -f1 VALIDATION_SET.tmp > true.tmp
                cut -f1 OUTPUT.tmp-predictions.tsv > pred.tmp
                paste -d'\t' true.tmp pred.tmp > true_vs_pred.tsv
                head true_vs_pred.tsv
                R
                df_mlp = read.table("true_vs_pred.tsv", T)
                str(df_mlp)
                cor(df_mlp)

                df_training = read.table("TRAINING_SET.tmp", T)
                df_validation = read.table("VALIDATION_SET.tmp", T)
                X = as.matrix(df_training[, 2:ncol(df_training)])
                y = df_training[, 1]
                
                # OLS
                b_hat = t(X) %*% solve(X %*% t(X)) %*% y
                y_hat = as.matrix(df_validation[, 2:ncol(df_validation)]) %*% b_hat
                cor(df_validation[, 1], y_hat)

                # Bayesian models
                library(BGLR)
                df = rbind(df_training, df_validation)
                X = as.matrix(df[, 2:ncol(df)])
                y = df[, 1]
                yNA = y
                idx_validation = nrow(df_training):nrow(df)
                yNA[idx_validation] = NA
                nIter=6000; burnIn=1000

                # Bayes A
                time = Sys.time()
                fmBA=BGLR(y=yNA,ETA=list( list(X=X,model='BayesA')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
                print(Sys.time() - t)
                yHat=fmBA$yHat[idx_validation]
                cor(yHat, y[idx_validation])

                # Bayes B
                time = Sys.time()
                fmBB=BGLR(y=yNA,ETA=list( list(X=X,model='BayesB')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
                print(Sys.time() - t)
                yHat=fmBB$yHat[idx_validation]
                cor(yHat, y[idx_validation])

                # Bayes C
                time = Sys.time()
                fmBC=BGLR(y=yNA,ETA=list( list(X=X,model='BayesC')), nIter=nIter,burnIn=burnIn,saveAt='ba_')
                print(Sys.time() - t)
                yHat=fmBC$yHat[idx_validation]
                cor(yHat, y[idx_validation])
            done
        done
    done
done




```

# Remote-sensing modelling

## Empirical data test only (as simulating data will not be that different from the continuous simulated data in GP)

Potential dta sources:
- https://datadryad.org/dataset/doi:10.5061/dryad.v41ns1s4z
- https://datadryad.org/dataset/doi:10.5061/dryad.r4xgxd2mz

# Miscellaneous

## MLPInterrogator.jl

Planning on making an `mlp` model interrogator, i.e. an interactive Julia library to interrogate the model output (`*.json*`).
I anticipate users to ask: What if I want to see how some of the same categorical factor levels (e.g. a subset of entries in a yield trial) perform under some other categorical factor levels or continuous explanatory variable values they did not have empirical observations on (e.g. on a different set of environments)? --> Now, that I've written this down, the `--predict-only` flag does this. However, the new input feature data needs to be generated by-hand and so this interactive Julia tool is better poised to just generate these new sets of input data sets (with dummy target values set as NAN) for use with `mlp`!

