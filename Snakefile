import os, glob

MLP = "path/to/mlp_executable"
ROOT_DIR = "path/to/tmp"
DATA_TYPES = ["trials", "gp"]
N_YEARS = [2, 5]
N_SITES = [1, 3]
N_TREATMENTS = [1, 3]
N_ENTRIES = [25, 50]
N_REPLICATIONS = [3, 10]
N_HIDDEN_LAYERS = [1, 2, 3]

TRIALS_OUTPUT = expand(
    "{ROOT_DIR}/{DATA_TYPES}/input_simulated-YEARS_${N_YEARS}-SITES_${N_SITES}-TREATMENTS_${N_TREATMENTS}-ENTRIES_${N_ENTRIES}-REPLICATIONS_${N_REPLICATIONS}-HIDDEN_LAYERS_${N_HIDDEN_LAYERS}.tsv",
    ROOT_DIR=ROOT_DIR,
    DATA_TYPES=DATA_TYPES,
    N_YEARS=N_YEARS, N_SITES=N_SITES, N_TREATMENTS=N_TREATMENTS, N_ENTRIES=N_ENTRIES, N_REPLICATIONS=N_REPLICATIONS, N_HIDDEN_LAYERS=N_HIDDEN_LAYERS
)

rule all:
    input:
        TRIALS_OUTPUT

rule simulate_trials:
    output:
        "results/sim-{years}-{sites}-{treatments}-{entries}-{replications}-{hidden-layers}.done"
    params:
        mlp=MLP,
        root_dir=ROOT_DIR
    log:
        "logs/simulate_trials-{years}-{sites}-{treatments}-{entries}-{replications}-{hidden-layers}.log"
    conda:
        "conda.yaml"
    shell:
        """
        time \
        sh simulate.sh \
            {params.mlp} \
            trials \
            {params.root_dir} \
            {wildcards.years} \
            {wildcards.sites} \
            {wildcards.treatments} \
            {wildcards.entries} \
            {wildcards.replications} \
            {wildcards.hidden_layers}
        
        """
