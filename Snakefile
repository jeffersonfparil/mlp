import os, glob

MLP = "path/to/mlp_executable"
ROOT_DIR = "path/to/tmp/"

# ranges for numeric args
N_YEARS = [2, 5]
N_SITES = [1, 3]
N_TREATMENTS = [1, 3]
N_ENTRIES = [25, 50]
N_REPLICATIONS = [3, 10]
N_HIDDEN_LAYERS = [1, 2, 3]

ALL_RUNS = expand(
    "results/sim-{years}-{sites}-{treatments}-{entries}-{replications}-{hidden-layers}.done",
    years=N_YEARS, sites=N_SITES, treatments=N_TREATMENTS, entries=N_ENTRIES, replications=N_REPLICATIONS, hidden_layers=N_HIDDEN_LAYERS
)

rule all:
    input:
        ALL_RUNS

rule simulate_trials:
    output:
        "results/sim-{years}-{sites}-{treatments}-{entries}-{replications}-{hidden-layers}.done"
    params:
        mlp=MLP,
        root_dir=ROOT_DIR
    log:
        "logs/simulate_trials-{years}-{sites}-{treatments}-{entries}-{replications}-{hidden-layers}.log"
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
