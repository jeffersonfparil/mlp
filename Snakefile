from pathlib import Path

ROOT_OUTDIR = str(Path.home() / "Documents/mlp/tests/scripts/tmp")
SCRIPTS_DIR = str(Path.home() / "Documents/mlp/tests/scripts")
MLP = str(Path.home() / "Documents/mlp/target/release/mlp")
ANALYSIS_TYPES = ["trials", "gp"]
N_YEARS = [1, 3]
N_SITES = [1, 5]
N_TREATMENTS = [1, 3]
N_ENTRIES = [10, 50]
N_REPLICATIONS = [3]
N_HIDDEN_LAYERS = [1, 2]
DATA_TYPES = ["CONTINUOUS", "BINARY"]
N_OBSERVATIONS = [700]
N_FEATURES = [42000]

TRIALS_EMPIRICAL_INPUT = ["australia.soybean.txt", ]

TRIALS_SIMULATED_INPUT = expand(
    "{root_outdir}/trials/input_simulated-YEARS_{year}-SITES_{site}-TREATMENTS_{treatment}-ENTRIES_{entry}-REPLICATIONS_{replication}-HIDDEN_LAYERS_{hidden_layer}.tsv",
    root_outdir=ROOT_OUTDIR,
    year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
)

GP_SIMULATED_INPUT = expand(
    "{root_outdir}/gp/input_simulated-DATA_TYPE_{data_type}-N_{n}-P_{p}-HIDDEN_LAYERS_{hidden_layers}.tsv",
    root_outdir=ROOT_OUTDIR,
    data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
)

TRIALS_EMPIRICAL_INPUT = expand(
    # TODO: issue is we will have a variable number of empirical input files for each empirical dataset, e.g.:
    #   - australia.soybean.txt:
    #       + australia.soybean-yield.tsv
    #       + australia.soybean-height.tsv
    #       + australia.soybean-lodging.tsv
    #       + australia.soybean-size.tsv
    #       + australia.soybean-protein.tsv
    #       + australia.soybean-oil.tsv
    #   - ilri.sheep.txt:
    #       + ilri.sheep-birthwt.tsv
    #       + ilri.sheep-weanwt.tsv
    #       + ilri.sheep-weanage.tsv
)

rule all:
    input:
        TRIALS_SIMULATED_INPUT,
        GP_SIMULATED_INPUT,

rule simulate_trials:
    output:
        "{root_outdir}/trials/input_simulated-YEARS_{year}-SITES_{site}-TREATMENTS_{treatment}-ENTRIES_{entry}-REPLICATIONS_{replication}-HIDDEN_LAYERS_{hidden_layer}.tsv"
    params:
        scripts_dir=SCRIPTS_DIR,
        mlp=MLP,
    log:
        "{root_outdir}/trials/simulate_trials-YEARS_{year}-SITES_{site}-TREATMENTS_{treatment}-ENTRIES_{entry}-REPLICATIONS_{replication}-HIDDEN_LAYERS_{hidden_layer}.log"
    conda:
        "conda.yaml"
    shell:
        """
        cd {wildcards.root_outdir}
        time \
        sh {params.scripts_dir}/simulate.sh \
            {params.mlp} \
            trials \
            {wildcards.root_outdir}/trials \
            {wildcards.year} \
            {wildcards.site} \
            {wildcards.treatment} \
            {wildcards.entry} \
            {wildcards.replication} \
            {wildcards.hidden_layer} > {log}
        """

rule simulate_gp:
    output:
        "{root_outdir}/gp/input_simulated-DATA_TYPE_{data_type}-N_{n}-P_{p}-HIDDEN_LAYERS_{hidden_layers}.tsv"
    params:
        scripts_dir=SCRIPTS_DIR,
        mlp=MLP,
    log:
        "{root_outdir}/gp/simulate_gp-DATA_TYPE_{data_type}-N_{n}-P_{p}-HIDDEN_LAYERS_{hidden_layers}.log"
    conda:
        "conda.yaml"
    shell:
        """
        cd {wildcards.root_outdir}
        time \
        sh {params.scripts_dir}/simulate.sh \
            {params.mlp} \
            gp \
            {wildcards.root_outdir}/gp \
            {wildcards.data_type} \
            {wildcards.n} \
            {wildcards.p} \
            {wildcards.hidden_layers} > {log}
        """

rule empiricalprep_trials:
    output:
    params:
    log:
    conda:
        "conda.yaml"
    shell:
        """
        cd {wildcards.root_outdir}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            trials \
            ??? > {log}
        """