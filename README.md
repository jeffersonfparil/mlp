# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

# Quickstart

```shell
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
          Activation function (Choose from: "ReLU", "Sigmoid", "HyperbolicTangent", "Linear") (Note: "LeakyReLU" under construction) [default: ReLU]
      --cost <COST>
          Cost function (Choose: "MSE", "MAE", "HL") [default: MSE]
      --optimiser <OPTIMISER>
          Optimiser (Choose: "Adam", "AdamMax", "GradientDescent") [default: Adam]
      --n-epochs <N_EPOCHS>
          Maximum number of training epochs [default: 10]
      --n-burnin-epochs <N_BURNIN_EPOCHS>
          Number of burnin epochs (initial training epochs to discard) [default: 0]
      --f-patient-epochs <F_PATIENT_EPOCHS>
          Fraction of the maximum number of epochs to wait before enabling the criteria for early stopping [default: 0.25]
      --f-validation <F_VALIDATION>
          Fraction of the observations to be used in the estimation of cost at every epoch (using a fixed set of randomly chosen [seeded] observations across all epochs) [default: 0]
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
          Range of learning rates for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 1e-3,1e-3,1e-3]
      --range-n-epochs <RANGE_N_EPOCHS>
          Range of maximum number of training epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 10,10,10]
      --range-n-burnin-epochs <RANGE_N_BURNIN_EPOCHS>
          Range of burnin epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 0,1,1]
      --range-f-patient-epochs <RANGE_F_PATIENT_EPOCHS>
          Range of proportions of the maximum training epochs to start considering early stopping for hyperparameter optimisation (elements correspond to minimum, maximum and step size) [default: 0.5,1.0,0.5]
      --range-f-validation <RANGE_F_VALIDATION>
          Range of proportions of the observations to be used in within training validation set [default: 0.0,0.25,0.01]
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
time cargo run -- -s -n100 -p2000 -q0 -v
# time cargo run -- -s -n500 -p12000 -q0 -v # 13 minutes on gpu001: Intel(R) Xeon(R) Gold 5418Y 24 cores with 1 NVIDIA H100 NVL (93.584Gi)
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

# Field trial analysis

We want to show that MLP estimates genotype effects similar to those of linear models while also getting better model fit.

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

Run `tests/scripts/trials-simulate.sh`:

```shell
cd mlp/
mkdir tests/trials
mkdir tests/trials/simulated
cd tests/trials/simulated
time sh ../../scripts/trials-simulate.sh
# real    1m2.264s
# user    0m2.556s
# sys     1m1.800s
```

### Analysis using R

Run `tests/scripts/trials-linear_fit.R`:

```shell
cd mlp
cd tests/trials/simulated
module load ASReml-R # if ASReml-R is available
time Rscript ../../scripts/trials-linear_fit.R
# real    3m22.916s
# user    61m14.510s
# sys     0m12.706s
```

### Analysis using mlp

Run `tests/scripts/trials-mlp_fit.sh`:

```shell
cd mlp
cd tests/trials/simulated
time sh ../../scripts/trials-mlp_fit.sh
# real    1m46.854s
# user    0m47.295s
# sys     1m4.848s
```

### Comparison between linear mixed model and mlp

Run `tests/scripts/trials-comparisons.R`:

```shell
cd mlp
cd tests/trials/simulated
time Rscript ../../scripts/trials-comparisons.R
# real    0m0.395s
# user    0m6.552s
# sys     0m0.062s
```

</details>

### Results

#### SMALL-1HL

![](./tests/trials/simulated/comparison-SMALL-1HL.png)

#### SMALL-2HL

![](./tests/trials/simulated/comparison-SMALL-2HL.png)

#### SMALL-3HL

![](./tests/trials/simulated/comparison-SMALL-3HL.png)

#### SMALL-4HL

![](./tests/trials/simulated/comparison-SMALL-4HL.png)

#### SMALL-5HL

![](./tests/trials/simulated/comparison-SMALL-5HL.png)

#### LARGE-1HL

![](./tests/trials/simulated/comparison-LARGE-1HL.png)

#### LARGE-2HL

![](./tests/trials/simulated/comparison-LARGE-2HL.png)

#### LARGE-3HL

![](./tests/trials/simulated/comparison-LARGE-3HL.png)

#### LARGE-4HL

![](./tests/trials/simulated/comparison-LARGE-4HL.png)

#### LARGE-5HL

![](./tests/trials/simulated/comparison-LARGE-5HL.png)

## Tests on empirical data

We use the [agridat](https://kwstat.github.io/agridat/) dataset and for simplicity we only inclulded replicated datasets and with the `gen` factor. We analyse each trait separately.

<details>

### Prepare test data

- Explanatory variables can be numeric or categorical which means that explanatory variables which are written as numeric but are meant to be categorical need to be converted into strings, e.g. convert `rep=[1, 1, 1, 2, 2, 2, 3, 3]` into `rep=["rep➵1", "rep➵1", "rep➵1", "rep➵2", "rep➵2", "rep➵2", "rep➵3", "rep➵3"]`.
- We exclude covariance and uniformity datasets.
- We only include datasets with the genotype (`gen`) variable because we are testing `mlp`'s applicability for breeding, i.e. we want to rank the genotypes for selection.
- We then generate one file with a single response variable named `y` in the first column followed by the explanatory variables.
- We remove rows with at least one missing data.
- We also exclude datasets with unreplicated `gen` factors.

Download and prepare the empirical datasets:

```shell
cd mlp/
mkdir tests/trials/empirical/
cd tests/trials/empirical/
time sh ../../scripts/trials-download_empirical.sh
# real    0m5.563s
# user    0m1.003s
# sys     0m0.509s
time Rscript ../../scripts/trials-prepare_empirical.R
# real    0m1.390s
# user    0m1.099s
# sys     0m0.202s
```

### Analysis using R

Run `tests/scripts/trials-linear_fit.R`:

```shell
cd mlp
cd tests/trials/empirical/
module load ASReml-R # if ASReml-R is available
time Rscript ../../scripts/trials-linear_fit.R
# ???? test run 20260507
```

Using slurm because this will take a long time probably --> Run `tests/trials/empirical/trials-linear_fit.slurm`:

```slurm
#!/bin/bash
#SBATCH --job-name="mlp"
#SBATCH --account="dbiof1"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=5-0:0:00
module load ASReml-R # if ASReml-R is available
time Rscript ../../scripts/trials-linear_fit.R
##############################################
# cd tests/trials/empirical/
# wc -l slurm-16646012.out
# tail slurm-16646012.out
# grep -A1 "@@@" slurm-16646012.out | tail -n1
# cat slurm-16646012.out
```

### Analysis using mlp

Run `tests/scripts/trials-mlp_fit.sh`:

```shell
cd mlp
cd tests/trials/empirical/
time sh ../../scripts/trials-mlp_fit.sh
# real    23m27.114s
# user    16m41.525s
# sys     7m0.550s
```

### Comparison between linear mixed model and mlp

Run `tests/scripts/trials-comparisons.R`:

```shell
cd mlp
cd tests/trials/empirical/
time Rscript ../../scripts/trials-comparisons.R
# real    0m2.732s
# user    0m8.683s
# sys     0m0.146s
```

</details>

### Results

#### ars.earlywhitecorn96-yield

![](./tests/trials/empirical/comparison-ars.earlywhitecorn96-yield.png)

#### australia.soybean-yield

![](./tests/trials/empirical/comparison-australia.soybean-yield.png)

#### barrero.maize-daystoflower

![](./tests/trials/empirical/comparison-barrero.maize-daystoflower.png)


# Genomic prediction

We want to show that MLP performs as good but mostly better than Bayesian linear models for genomic prediction.
We evaluated performances based on 10-fold cross-validation repeated 5 times:
    - simulated relatively small datasets (i.e. 700 entries x 42,000 loci which may be continuous allele frequencies or binary genotypes)
    - empirical dataset from [Azodi et al (2019)](https://datadryad.org/dataset/doi:10.5061/dryad.xksn02vb9)

*Notes:* 
- We will need to increase the number of MCMC iterations for the BGLR to more realistic level, i.e. from 1,500 (500 burin) to 6,000 (1,000 burnin) iterations.
- We may also opt to perform hyperparameter optimisation for mlp, instead of the fixed parameters below:
    + ... **TODO UPDATE: testing hyperparam optim choosing betwween ReLU vs Liner activations** ...
    + ... or maybe use a simple parameter adjustment depending on the number of loci, e.g. increase the number of hidden nodes to 2048 if the number of loci drops below 1,000?

## Tests on simulated data

<details>

### Simulate data

We simulated:
- 700 entries
- 42,000 loci
- 2 types of datasets:
  + continuous allele frequencies ranging from 0 to 1
  + binary genotypes, i.e. 0 or 1
- With 1 to 2 hidden layers (no larger number of hidden layers because of GPU memory limitations in the H100s and V100s we have access to)

Run `tests/scripts/gp-simulate.sh`:

```shell
cd ${HOME}/Documents/mlp
mkdir tests/gp
mkdir tests/gp/simulated
cd tests/gp/simulated
time sh ../../scripts/gp-simulate.sh
# real    3m6.970s
# user    2m53.788s
# sys     0m13.480s
```

### GP CV using BGLR

Run `tests/scripts/gp-linear_fit.slurm`:

```shell
DIR=${HOME}/Documents/mlp
DIR_SRC=${DIR}/tests/scripts
DIR_DATA=${DIR}/tests/gp/simulated
cd $DIR_DATA
sbatch --array=1-$(ls input_simulated-*.tsv | wc -l) ${DIR_SRC}/gp-linear_fit.slurm ${DIR_SRC} ${DIR_DATA}
# grep -n -i "err" slurm-*.out
# grep -n -i "skip" slurm-*.out
# tail slurm-*.out
# head -n100 output_simulated-*.tsv
```

### MLP

Run `tests/scripts/gp-mlp_fit.slurm`:

```shell
DIR=${HOME}/Documents/mlp
MLP=${DIR}/target/release/mlp
DIR_SRC=${DIR}/tests/scripts
DIR_DATA=${DIR}/tests/gp/simulated
cd $DIR_DATA
mkdir ${DIR_DATA}/mlp_misc_output
sbatch --array=1-$(ls input_simulated-*.tsv | wc -l) ${DIR_SRC}/gp-mlp_fit.slurm ${MLP} ${DIR_SRC} ${DIR_DATA}
# grep -n -i "err" slurm-*.out
# grep -n -i "skip" slurm-*.out
# tail slurm-*.out
# head -n100 output_simulated-*.tsv
```

### Comparison between linear mixed model and mlp

Run `tests/scripts/trials-comparisons.R`:

```shell
DIR=${HOME}/Documents/mlp
cd ${DIR}/tests/gp/simulated
time Rscript ${DIR}/tests/scripts/gp-comparisons.R "gp-comparisons-simulated.png"
# real    0m0.324s
# user    0m4.892s
# sys     0m0.088s
```

</details>

### Results

![](./tests/gp/simulated/gp-comparisons-simulated.png)


## Tests on empirical data

<details>

### Prepare the data

Drayd has bot protection guards which means we need to download the Azodi et al (2019) data manually from here [https://datadryad.org/dataset/doi:10.5061/dryad.xksn02vb9](https://datadryad.org/dataset/doi:10.5061/dryad.xksn02vb9).

Extract the datasets into `tests/gp/empirical`, then merge the phenotype and genotype data per species such that we have one the trait/response variable at the first column and then the marker data for the rest of the columns, and yield 1 `.tsv` file per trait across datasets/species.

Run `tests/scripts/gp-prepare_empirical.sh`:

```shell
DIR=${HOME}/Documents/mlp
DIR_DATA=${DIR}/tests/gp/empirical
cd $DIR_DATA
time sh ${DIR}/tests/scripts/gp-prepare_empirical.sh
# real    2m10.783s
# user    2m6.665s
# sys     0m5.534s
```

### GP CV using BGLR

Run `tests/scripts/gp-linear_fit.slurm`:

```shell
DIR=${HOME}/Documents/mlp
DIR_SRC=${DIR}/tests/scripts
DIR_DATA=${DIR}/tests/gp/empirical
cd $DIR_DATA
sbatch --array=1-$(ls *.tsv | grep -v "output" | wc -l) ${DIR_SRC}/gp-linear_fit.slurm ${DIR_SRC} ${DIR_DATA}
# grep -n -i "err" slurm-*.out
# grep -n -i "skip" slurm-*.out
# tail slurm-*.out
# head -n100 output-*LINEAR*.tsv
```

### MLP

Run `tests/scripts/gp-mlp_fit.slurm`:

```shell
DIR=${HOME}/Documents/mlp
MLP=${DIR}/target/release/mlp
DIR_SRC=${DIR}/tests/scripts
DIR_DATA=${DIR}/tests/gp/empirical
cd $DIR_DATA
mkdir ${DIR_DATA}/mlp_misc_output
# sbatch --array=1-$(ls *.tsv | grep -v "output" | wc -l) ${DIR_SRC}/gp-mlp_fit.slurm ${MLP} ${DIR_SRC} ${DIR_DATA}
sbatch --array=4-15 ${DIR_SRC}/gp-mlp_fit.slurm ${MLP} ${DIR_SRC} ${DIR_DATA} # smaller datasets (below 100k loci)
sbatch --array=1-3 ${DIR_SRC}/gp-mlp_fit.slurm ${MLP} ${DIR_SRC} ${DIR_DATA} # large maize datasets
sbatch --array=16-18 ${DIR_SRC}/gp-mlp_fit.slurm ${MLP} ${DIR_SRC} ${DIR_DATA} # large switchgrass datasets
# grep -n -i "err" slurm-*.out
# grep -n "Activation:" slurm-*.out
# tail slurm-*.out
# head -n100 output-*MLP*.tsv
```

### Comparison between linear mixed model and mlp

Run `tests/scripts/trials-comparisons.R`:

```shell
DIR=${HOME}/Documents/mlp
cd ${DIR}/tests/gp/empirical
time Rscript ${DIR}/tests/scripts/gp-comparisons.R "gp-comparisons-empirical.png"
# real    0m0.464s
# user    0m4.964s
# sys     0m0.106s
```

</details>

### Results

Tests on empirical data running...

![](./tests/gp/empirical/gp-comparisons-empirical.png)



# Remote-sensing modelling

Empirical data test only (as simulating data will not be that different from the continuous simulated data in GP)

<details>

## Prepare the dataset

Download dataset from [Fared et al (2024)](https://datadryad.org/dataset/doi:10.5061/dryad.v41ns1s4z).



## Fit using canonical methods --> python stuff??

## MLP

</details>

# Miscellaneous

<details>

## MLPInterrogator.jl

Planning on making an `mlp` model interrogator, i.e. an interactive Julia library to interrogate the model output (`*.json*`).
I anticipate users to ask: What if I want to see how some of the same categorical factor levels (e.g. a subset of entries in a yield trial) perform under some other categorical factor levels or continuous explanatory variable values they did not have empirical observations on (e.g. on a different set of environments)? --> Now, that I've written this down, the `--predict-only` flag does this. However, the new input feature data needs to be generated by-hand and so this interactive Julia tool is better poised to just generate these new sets of input data sets (with dummy target values set as NAN) for use with `mlp`!

</details>
