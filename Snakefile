import os, glob
from pathlib import Path

ROOT_OUTDIR = str(Path.home() / "Documents/mlp/tests/scripts/tmp")
SCRIPTS_DIR = str(Path.home() / "Documents/mlp/tests/scripts")
MLP = str(Path.home() / "Documents/mlp/target/release/mlp")
DATA_TYPES = ["trials", "gp"]
N_YEARS = [2, 5]
N_SITES = [1, 3]
N_TREATMENTS = [1, 3]
N_ENTRIES = [25, 50]
N_REPLICATIONS = [3, 10]
N_HIDDEN_LAYERS = [1, 2, 3]

TRIALS_OUTPUT = expand(
    "{root_outdir}/{data_type}/input_simulated-YEARS_{year}-SITES_{site}-TREATMENTS_{treatment}-ENTRIES_{entry}-REPLICATIONS_{replication}-HIDDEN_LAYERS_{hidden_layer}.tsv",
    root_outdir=ROOT_OUTDIR,
    data_type=DATA_TYPES,
    year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
)

rule all:
    input:
        TRIALS_OUTPUT

rule simulate_trials:
    output:
        # "{root_outdir}/{data_type}/sim-{year}-{site}-{treatment}-{entry}-{replication}-{hidden_layer}.done"
        "{root_outdir}/{data_type}/input_simulated-YEARS_{year}-SITES_{site}-TREATMENTS_{treatment}-ENTRIES_{entry}-REPLICATIONS_{replication}-HIDDEN_LAYERS_{hidden_layer}.tsv"
    params:
        scripts_dir=SCRIPTS_DIR,
        mlp=MLP,
    log:
        "{root_outdir}/{data_type}/simulate_trials-{year}-{site}-{treatment}-{entry}-{replication}-{hidden_layer}.log"
    conda:
        "conda.yaml"
    shell:
        """
        time \
        sh {params.scripts_dir}/simulate.sh \
            {params.mlp} \
            trials \
            {wildcards.root_outdir} \
            {wildcards.year} \
            {wildcards.site} \
            {wildcards.treatment} \
            {wildcards.entry} \
            {wildcards.replication} \
            {wildcards.hidden_layer}
        
        """
