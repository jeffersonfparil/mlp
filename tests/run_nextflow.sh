#!/bin/bash
#SBATCH --job-name=mlp_benchmarking
#SBATCH --account=dbiof2
#SBATCH --partition=batch
#SBATCH --time=28-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/mlp_benchmarking_%j.out
#SBATCH --error=logs/mlp_benchmarking_%j.err

module load Nextflow/25.10.4

WORKDIR=${HOME}/Documents/mlp/tests
cd $WORKDIR

nextflow run nextflow_pipeline/main.nf \
    -c nextflow_pipeline/params.config \
    -c nextflow_pipeline/process.config \
    -resume