#!/bin/bash
#SBATCH --job-name="mlp"
#SBATCH --account="dbiof1"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=50G
#SBATCH --time=5-0:0:00
# cd ${HOME}/Documents/mlp; pixi shell

# TODO: update these to be more similar to `gp-linear_fit.slurm`...

MLP=${HOME}/Documents/mlp/target/release/mlp
DIR=${HOME}/Documents/mlp/tests/gp/simulated
SCRIPT=${HOME}/Documents/mlp/tests/scripts/gp-mlp_fit.sh
INPUT=$(find "$DIR" -wholename "$DIR/input_simulated-*.tsv" | head -n${SLURM_ARRAY_TASK_ID} | tail -n1)
time sh $SCRIPT $INPUT $MLP $DIR
