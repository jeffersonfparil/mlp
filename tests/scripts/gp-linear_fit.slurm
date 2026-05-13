#!/bin/bash
#SBATCH --job-name="mlp"
#SBATCH --account="dbiof1"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=5-0:0:00
DIR_SRC=$1
DIR_DATA=$2
FNAME_INPUT=$(find $DIR_DATA -wholename "*.tsv" | grep -v "output" | sort | head -n${SLURM_ARRAY_TASK_ID} | tail -n1)
time Rscript ${DIR_SRC}/gp-linear_fit.R ${FNAME_INPUT}
