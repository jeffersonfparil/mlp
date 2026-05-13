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
MLP=$1
DIR_SRC=$2
DIR_DATA=$3
INPUT=$(find "$DIR_DATA" -wholename "$DIR_DATA/input_simulated-*.tsv" | head -n${SLURM_ARRAY_TASK_ID} | tail -n1)
time sh ${DIR_SRC}/gp-mlp_fit.sh $INPUT $MLP $DIR_DATA
