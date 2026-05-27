# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

# Quickstart

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
time cargo run -- -s -n100 -p2000 -q0 -v
# time cargo run -- -s -n1k -p12000 -q0 -v # 13 minutes on gpu001: Intel(R) Xeon(R) Gold 5418Y 24 cores with 1 NVIDIA H100 NVL (93.584Gi)
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


# time cargo run -- -s -n1000 -p10 -v
# INPUT=$(ls -t1 | grep "input.*.tsv" | head -n1)
# time cargo run -- \
#    -f $INPUT \
#    -o OUTPUT.tmp.json \
#    -v \
#    --hyperparameter-optimisation \
#    --range-hidden-layers="1,1,1" \
#    --range-hidden-layer-nodes="700,700,700" \
#    --range-dropout-rates="0.0,0.0,0.01" \
#    --range-learning-rates="1e-5,1e-5,1e-5" \
#    --range-n-epochs="1000,1000,1000" \
#    --range-n-burnin-epochs="100,100,100" \
#    --range-f-patient-epochs="0.01,0.01,0.01" \
#    --range-f-validation="0.1,0.1,0.1" \
#    --range-n-batches="1,1,1" \
#    --selection-costs="MSE" \
#    --selection-optimisers="Adam,GradientDescent" \
#    --selection-activations="ReLU,Linear" \
#    --selection-weights-initialisations="He,Cauchy" \
#    --skip-marginals

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

# Systematic testing workflow using Snakemake

```shell
cd mlp/
pixi shell
snakemake --lint
snakemake -n # or --dry-run
N_CORES=24
# N_CORES=3
time snakemake --cores $N_CORES
# time pixi run snakemake --cores $N_CORES
```

# TODO: Remote-sensing modelling

- Empirical data test only (as simulating data will not be that different from the continuous simulated data in GP)
- Download dataset from [Fared et al (2024)](https://datadryad.org/dataset/doi:10.5061/dryad.v41ns1s4z).
- Fit using canonical methods --> python stuff??


# Miscellaneous

<details>

## MLPInterrogator.jl

Planning on making an `mlp` model interrogator, i.e. an interactive Julia library to interrogate the model output (`*.json*`).
I anticipate users to ask: What if I want to see how some of the same categorical factor levels (e.g. a subset of entries in a yield trial) perform under some other categorical factor levels or continuous explanatory variable values they did not have empirical observations on (e.g. on a different set of environments)? --> Now, that I've written this down, the `--predict-only` flag does this. However, the new input feature data needs to be generated by-hand and so this interactive Julia tool is better poised to just generate these new sets of input data sets (with dummy target values set as NAN) for use with `mlp`!

</details>
