# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

# Quickstart

## Installation

Download the binary compatible with your hardware:

- [Linux x86 with NVIDIA H100](https://github.com/jeffersonfparil/mlp/releases/download/v0.2.0/mlp-h100)
- [Linux x86 with NVIDIA NVIDIA Tesla V100](https://github.com/jeffersonfparil/mlp/releases/download/v0.2.0/mlp-v100)
- [Linux x86 with NVIDIA GeForce 940MX](https://github.com/jeffersonfparil/mlp/releases/download/v0.2.0/mlp-940mx)

This list is severely limited at the moment, but please feel free to build it from source. The [instructions are found below](#build-from-source-code).

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

## Install Nextflow (if not on an HPC or just wants to use pixi)

```shell
pixi global install -c conda-forge -c bioconda nextflow
```

## Run the workflow:

### On a single machine or one HPC node:
```shell
cd mlp/tests
pixi run nextflow run nextflow_pipeline/main.nf \
    -c nextflow_pipeline/params.config \
    -resume

# SMALL TEST
pixi run nextflow run nextflow_pipeline/main.nf \
    -c nextflow_pipeline/params_test.config \
    -resume \
    -with-trace trace.txt \
    -with-report report.html \
    -with-timeline timeline.txt \
    -with-dag flowchart.dot

```

### On an HPC system with Slurm

```shell
cd mlp/tests
sbatch run_nextflow.sh
```

# Miscellaneous

## MLPInterrogator.jl

<details>

Planning on making an `mlp` model interrogator, i.e. an interactive Julia library to interrogate the model output (`*.json*`).
I anticipate users to ask: What if I want to see how some of the same categorical factor levels (e.g. a subset of entries in a yield trial) perform under some other categorical factor levels or continuous explanatory variable values they did not have empirical observations on (e.g. on a different set of environments)? --> Now, that I've written this down, the `--predict-only` flag does this. However, the new input feature data needs to be generated by-hand and so this interactive Julia tool is better poised to just generate these new sets of input data sets (with dummy target values set as NAN) for use with `mlp`!

</details>

## Simulate an HPC

Under construction...

<details>

### Install slurm:

```shell
sudo apt install slurmd slurmctld -y
sudo chmod 777 /etc/slurm
sudo cat << EOF > /etc/slurm/slurm.conf
# slurm.conf file generated by configurator.html.
# Put this file on all nodes of your cluster.
# See the slurm.conf man page for more information.
#
ClusterName=localcluster
SlurmctldHost=localhost
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/lib/slurm/slurmd
SlurmUser=slurm
StateSaveLocation=/var/lib/slurm/slurmctld
SwitchType=switch/none
TaskPlugin=task/none
#
# TIMERS
InactiveLimit=0
KillWait=30
MinJobAge=300
SlurmctldTimeout=120
SlurmdTimeout=300
Waittime=0
# SCHEDULING
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
#
#AccountingStoragePort=
AccountingStorageType=accounting_storage/filetxt
JobCompType=jobcomp/none
JobAcctGatherFrequency=30
JobAcctGatherType=jobacct_gather/linux
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm/slurmd.log
AccountingStorageLoc=/var/log/slurm/accounting.txt
#
# COMPUTE NODES
NodeName=localhost CPUs=4 Boards=1 SocketsPerBoard=1 CoresPerSocket=2 ThreadsPerCore=2 RealMemory=15868 Gres=gpu:h100:1 State=UNKNOWN
PartitionName=gpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP
GresTypes=gpu
EOF
sudo chmod 755 /etc/slurm/
sudo systemctl start slurmctld
sudo systemctl start slurmd
sudo scontrol update nodename=localhost state=idle
sinfo
sudo cat /var/log/slurm/slurmd.log
sudo cat /var/log/slurm/slurmctld.log
```

### Install Lmod:

```shell
sudo apt install lua5.4 liblua5.4-dev lmod -y
sudo apt install tcl-dev -y
wget https://sourceforge.net/projects/lmod/files/Lmod-8.7.tar.bz2
tar xfvj Lmod-8.7.tar.bz2
rm Lmod-8.7.tar.bz2
cd Lmod-8.7/
./configure --prefix=$HOME --with-fastTCLInterp=no
sudo make install
echo 'export PATH=$HOME/lmod/8.7/libexec:$PATH' >> ~/.bashrc
echo 'source $HOME/lmod/8.7/init/bash' >> ~/.bashrc
echo 'export LMOD_CMD=$HOME/lmod/8.7/libexec/lmod' >> ~/.bashrc
echo 'export MODULEPATH="/etc/lmod/modules/"' >> ~/.bashrc
```

### Sample module file (/etc/lmod/modules/R.lua):

```shell
sudo chmod -R 777 /etc/lmod/modules/
sudo cat << EOF > /etc/lmod/modules/R.lua
help([[
...
]])
whatis("Version: 4.1.2")
whatis("R statistical computing environment")
prepend_path("LD_LIBRARY_PATH","/usr/local/lib/R/site-library/")
prepend_path("LIBRARY_PATH","\$HOME/R/x86_64-pc-linux-gnu-library/4.3")
prepend_path("PATH","/usr/bin")
EOF
sudo chmod -R 755 /etc/lmod/modules/
```

### sshare setup for snakemake-executor-plugin-slurm

```shell
cat << 'EOF' | sudo tee /usr/local/bin/sshare > /dev/null
#!/bin/bash
# Fake sshare data to satisfy Snakemake SLURM plugin
echo "Account|User|RawShares|NormShares|RawUsage|NormUsage|FairShare"
echo "localuser"
EOF
sudo chmod +x /usr/local/bin/sshare
```

### GPU setup

```shell
ls -lh /etc/slurm/
sudo touch /etc/slurm/fake_gpu
cat << 'EOF' | sudo tee /etc/slurm/gres.conf > /dev/null
NodeName=localhost Name=gpu Type=h100 Count=1 Flags=CountOnly
EOF
sudo chmod 644 /etc/slurm/gres.conf

sudo sed -i 's/localhost/paril-ThinkPad-T470/g' /etc/slurm/slurm.conf
sudo sed -i 's/localhost/paril-ThinkPad-T470/g' /etc/slurm/gres.conf

sudo systemctl restart slurmd
sudo systemctl restart slurmctld
sudo scontrol update NodeName=paril-ThinkPad-T470 State=DOWN Reason="name change reset"
sudo scontrol update NodeName=paril-ThinkPad-T470 State=RESUME

sudo tail -n 15 /var/log/slurm/slurmctld.log

sudo mknod -m 666 /dev/fake_nvidia c 195 255

cat << 'EOF' | sudo tee /etc/slurm/gres.conf > /dev/null
NodeName=paril-ThinkPad-T470 Name=gpu Type=h100 File=/dev/fake_nvidia
EOF

sudo systemctl restart slurmd
sudo systemctl restart slurmctld

sudo scontrol update NodeName=paril-ThinkPad-T470 State=DOWN Reason="hardware fix"
sudo scontrol update NodeName=paril-ThinkPad-T470 State=RESUME

sudo touch /var/log/slurm/accounting.txt
sudo chmod 777 /var/log/slurm/accounting.txt

sinfo
```

### Test

```shell
conda config --set channel_priority strict
pixi run snakemake --executor slurm --jobs 1 --use-conda --default-resources slurm_account="localuser"

module avail R
module add R
```

</details>
