#!/bin/bash
# properties = {properties}



module purge
module load Miniconda3/24.7.1-0
module load snakemake


# Debug (optional but useful)
which conda
conda info

ldd $(which python) | grep libpython

exec "$@"
