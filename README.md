# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

# Quickstart

## Installation

Download the binary compatible with your hardware:

- [Linux x86 with NVIDIA H100 or NVIDIA V100](https://github.com/jeffersonfparil/mlp/releases/download/v0.2.0/mlp)

This list is severely limited at the moment, but please feel free to build it from source. The [instructions are found below](#development-setup).

## Quick demo

Simulate data, fit, and extract marginal effects (`-v` for verbose):

```shell
./mlp -v
```

## Using default hyperparameters

Using the default hyperparamereters should suffice for trial analyses where the main objective is the estimation of the ranks of the entries, treatments, sites, etc.

1. Simulate a field trial dataset:

```shell
INPUT="input.tsv"
time \
./mlp \
    --simulate-data-only \
    --simulation-fname-output=$INPUT \
    --simulation-n-observations $(echo "2*3*5*6*10" | bc) \
    --simulation-n-features-continuous 0 \
    --simulation-n-features-categorical 2,3,5,6,10 \
    --simulation-n-output-columns 1 \
    --verbose
```

2. Fit without extracting marginal effects:

```shell
INPUT="input.tsv"
OUTPUT="output.json"
./mlp -f $INPUT -o $OUTPUT --skip-marginals
```

3. Extract marginal effects:

```shell
INPUT="input.tsv"
OUTPUT="output.json"
./mlp --marginals-only --model $OUTPUT -f $INPUT
cat ${OUTPUT%.json*}-marginal_effects.tsv
```

## Using fixed hyperparameters

Here we demonstrate fitting a dataset with more predictors than observations using fixed hyperparameters where our objective is prediction rather than description, i.e. we do not intend to estimate marginal effects.

1. Simulate a genomic prediction dataset:

```shell
INPUT="input_n1k.tsv"
time \
./mlp \
    --simulate-data-only \
    --simulation-fname-output=$INPUT \
    --simulation-n-observations 1000 \
    --simulation-n-features-continuous 23000 \
    --simulation-n-features-categorical 0 \
    --simulation-n-output-columns 1 \
    --simulation-n-hidden-layers 2 \
    --simulation-weights-distribution normal \
    --simulation-weights-distribution-param-1 0.0 \
    --simulation-weights-distribution-param-2 0.01 \
    --verbose
head  $INPUT | cut -f1-5
head -n901 $INPUT > ${INPUT%.tsv*}-TRAINING.tsv
head -n1 $INPUT > ${INPUT%.tsv*}-VALIDATION.tsv
tail -n100 $INPUT >> ${INPUT%.tsv*}-VALIDATION.tsv
```

2. Fit the training set without extracting marginal effects:
(**Note**: make sure to use the `--skip-marginals` flag because with large number of predictors, extracting marginal effects of each will take a long time)

```shell
INPUT="input_n1k-TRAINING.tsv"
OUTPUT="output_n1k.json"
time \
./mlp \
    -f $INPUT \
    -o $OUTPUT \
    --n-batches=1 \
    --n-hidden-layers=2 \
    --n-hidden-nodes=1024,128 \
    --n-epochs=1000 \
    --n-burnin-epochs=10 \
    --f-patient-epochs=0.1 \
    --f-validation=0.1 \
    --skip-marginals \
    --verbose
```

3. Predict the validation set

```shell
INPUT_TO_PREDICT="input_n1k-VALIDATION.tsv"
OUTPUT="output_n1k.json"
time \
./mlp \
    --predict-only \
    -f $INPUT_TO_PREDICT \
    --model $OUTPUT \
    --verbose
# Extract true and predicted values for assessment
PREDICTIONS=${OUTPUT%.json*}-predictions.tsv
cut -f1 $INPUT_TO_PREDICT > TRUE.tmp
cut -f1 $PREDICTIONS > PREDICTED.tmp
paste -d'\t' TRUE.tmp PREDICTED.tmp > TRUE_VS_PREDICTED.tsv
rm TRUE.tmp PREDICTED.tmp
```

4. Assess prediction:

```R
df = read.delim("TRUE_VS_PREDICTED.tsv", header=TRUE)
colnames(df) <- c("true", "predicted")
cor(df$true, df$predicted)
txtplot::txtplot(df$true, df$predicted)
```

## Using hyperparameter optimisation

We also demonstrate fitting a dataset with more predictors than observations but now using hyperparameter optimisation.

1. Simulate a genomic prediction dataset:

```shell
INPUT="input_n1k.tsv"
time \
./mlp \
    --simulate-data-only \
    --simulation-fname-output=$INPUT \
    --simulation-n-observations 1000 \
    --simulation-n-features-continuous 23000 \
    --simulation-n-features-categorical 0 \
    --simulation-n-output-columns 1 \
    --simulation-n-hidden-layers 2 \
    --simulation-weights-distribution normal \
    --simulation-weights-distribution-param-1 0.0 \
    --simulation-weights-distribution-param-2 0.01 \
    --verbose
head  $INPUT | cut -f1-5
head -n901 $INPUT > ${INPUT%.tsv*}-TRAINING.tsv
head -n1 $INPUT > ${INPUT%.tsv*}-VALIDATION.tsv
tail -n100 $INPUT >> ${INPUT%.tsv*}-VALIDATION.tsv
```

2. Fit the training set without extracting marginal effects:
(**Note**: again make sure to use the `--skip-marginals` flag because with large number of predictors, extracting marginal effects of each will take a long time)

```shell
INPUT="input_n1k-TRAINING.tsv"
OUTPUT="output_n1k.json"
time \
./mlp \
    -f $INPUT \
    -o $OUTPUT \
    --hyperparameter-optimisation \
    --range-hidden-layers=1,2,1 \
    --range-hidden-layer-nodes=100,1000,900 \
    --range-dropout-rates=0.0,0.0,0.01 \
    --range-learning-rates=1e-3,1e-3,1e-3 \
    --range-n-epochs=1000,1000,1000 \
    --range-n-burnin-epochs=10,10,10 \
    --range-f-patient-epochs=0.1,0.1,0.1 \
    --range-f-validation=0.0,0.2,0.1 \
    --range-n-batches=1,1,1 \
    --selection-activations=ReLU \
    --selection-costs=MSE \
    --selection-optimisers=Adam \
    --selection-weights-initialisations=He,Cauchy \
    --skip-marginals \
    --verbose
```

3. Predict the validation set

```shell
INPUT_TO_PREDICT="input_n1k-VALIDATION.tsv"
OUTPUT="output_n1k.json"
time \
./mlp \
    --predict-only \
    -f $INPUT_TO_PREDICT \
    --model $OUTPUT \
    --verbose
# Extract true and predicted values for assessment
PREDICTIONS=${OUTPUT%.json*}-predictions.tsv
cut -f1 $INPUT_TO_PREDICT > TRUE.tmp
cut -f1 $PREDICTIONS > PREDICTED.tmp
paste -d'\t' TRUE.tmp PREDICTED.tmp > TRUE_VS_PREDICTED.tsv
rm TRUE.tmp PREDICTED.tmp
```

4. Assess prediction:

```R
df = read.delim("TRUE_VS_PREDICTED.tsv", header=TRUE)
colnames(df) <- c("true", "predicted")
cor(df$true, df$predicted)
txtplot::txtplot(df$true, df$predicted)
```

# Build from source code

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

4. Build:

```shell
cd mlp
cargo build --release
./target/release/mlp -h
```

# Unit testing

```shell
cd mlp
pixi shell
# export LD_LIBRARY_PATH=${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib # in case the dynamic linker library is not in the path
time cargo test -- --show-output
```

# Data formats

1. Input data:
    - Default format is `TSV` (tab-delimited); other delimiters are supported via `-d` or `--delim`
    - The first column is assumed to contain numeric response values, but you may use one or more target columns anywhere in the file
    - Remaining columns are explanatory variables:
      + numeric → continuous or binary
      + non-numeric → categorical factor levels, converted to binary via one-hot encoding
    - See example: [`./misc/input_simulated-T20260527230243-R43101535.tsv`](./misc/input_simulated-T20260527230243-R43101535.tsv)

2. Model:
    - `JSON`: network model exported from the `Network` struct
    - `n_observations` (usize): number of observations
    - `n_features` (usize): number of input features
    - `n_targets` (usize): number of output dimensions
    - `n_hidden_layers` (usize): number of hidden layers
    - `n_hidden_nodes` (Vec<usize>): nodes per hidden layer
    - `dropout_rates` (Vec<f32>): dropout rate for each hidden layer
    - `targets` (Vec<f32>): observed values, standardised
    - `targets_mean_sd` (f32,f32): mean and standard deviation of targets
    - `predictions` (Vec<f32>): predicted values
    - `weights_per_layer` (Vec<Vec<f32>>): weight matrices by layer
    - `biases_per_layer` (Vec<Vec<f32>>): bias vectors by layer
    - `weights_x_biases_per_layer` (Vec<Vec<f32>>): pre-activation sums by layer
    - `activations_per_layer` (Vec<Vec<f32>>): layer outputs, including input layer
    - `weights_gradients_per_layer` (Vec<Vec<f32>>): weight gradients by layer
    - `biases_gradients_per_layer` (Vec<Vec<f32>>): bias gradients by layer
    - `activation` (String): activation function
    - `cost` (String): cost function
    - `weights_initialisation` (String): weights initialisation method (He, Cauchy, Uniform, StandardNormal)
    - `n_epochs` (usize): number of training epochs
    - `seed` (usize): random seed used for dropouts
    - `loss` (f32): mean loss (not part of the actual `Network` struct)
    - See example: [`./misc/output_network-T20260527230245-R616739134.json`](./misc/output_network-T20260527230245-R616739134.json)

3. Predictions: same as the input format, but response columns hold predicted values

4. Marginal effects:
    - `TSV`: tab-delimited
    - Estimates come from perturbation or SHAP methods
    - See example: [`./misc/output_network-T20260527230245-R616739134-marginal_effects.tsv`](./misc/output_network-T20260527230245-R616739134-marginal_effects.tsv)

5. Figures/plots:
    - `SVG`: loss curve and observed vs predicted scatterplot
    - `PNG`: marginal effects barplot
    - See examples: 
      + [`./misc/Loss_curve-ReLU-MSE-He-Adam-HL1-HN[128]-E1000-BE0-FPE0.01-FV0-B2-LR0.001-T20260527230245-R3399378659.svg`](./misc/Loss_curve-ReLU-MSE-He-Adam-HL1-HN[128]-E1000-BE0-FPE0.01-FV0-B2-LR0.001-T20260527230245-R3399378659.svg)
      + [`./misc/Observed_vs_predicted-ReLU-MSE-He-Adam-HL1-HN[128]-E1000-BE0-FPE0.01-FV0-B2-LR0.001-T20260527230245-R13995183.svg`](./misc/Observed_vs_predicted-ReLU-MSE-He-Adam-HL1-HN[128]-E1000-BE0-FPE0.01-FV0-B2-LR0.001-T20260527230245-R13995183.svg)
      + [`./misc/Marginal_effects-T20260527230245-R3240974675.png`](./misc/Marginal_effects-T20260527230245-R3240974675.png)

6. Special characters
    - Used in progress bars: `█`
    - Used as delimiters between non-numeric or categorical variable names and their levels: `➵`
    - Used as delimiters in marginals' combinations: `▓`

# Benchmarking

## Install Snakemake:

```shell
pixi global install snakemake conda -c conda-forge -c bioconda
```

## Run the workflow:

*Note*: comment-out the modelling part in `rule all` to use more compute cores and speed-up simulations and empirical data preparations, 
after which uncomment them and run with less cores to avoid running out of GPU memory.

### On a single machine:
```shell
cd mlp/
N_CORES=24
# N_CORES=2
time pixi run snakemake --cores $N_CORES --use-conda
# ### Debugging and development
# pixi shell
# snakemake --lint
# snakemake -n # or --dry-run
```

### On an HPC system with Slurm

```shell
cd mlp/
time pixi run snakemake --slurm --default-resources slurm_account="dbiof2" slurm_partition="cpu"
```

# Miscellaneous

<details>

## MLPInterrogator.jl

Planning on making an `mlp` model interrogator, i.e. an interactive Julia library to interrogate the model output (`*.json*`).
I anticipate users to ask: What if I want to see how some of the same categorical factor levels (e.g. a subset of entries in a yield trial) perform under some other categorical factor levels or continuous explanatory variable values they did not have empirical observations on (e.g. on a different set of environments)? --> Now, that I've written this down, the `--predict-only` flag does this. However, the new input feature data needs to be generated by-hand and so this interactive Julia tool is better poised to just generate these new sets of input data sets (with dummy target values set as NAN) for use with `mlp`!

</details>
