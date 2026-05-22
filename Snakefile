import os
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
TRIALS_AGRIDAT_DIR = str(Path.home() / "Documents/mlp/tests/datasets/agridat")
TRIALS_AGRIDAT_FNAMES = ["australia.soybean.txt", "ilri.sheep.txt"]

TRIALS_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/trials/input_simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.tsv",
    year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
)

GP_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/gp/input_simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.tsv",
    data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
)

TRIALS_EMPIRICAL_LOG = expand(
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
    f"{ROOT_OUTDIR}/trials/TMPDIR-{{fname}}/log",
    fname=TRIALS_AGRIDAT_FNAMES
)

rule all:
    input:
        TRIALS_SIMULATED_INPUT,
        GP_SIMULATED_INPUT,
        TRIALS_EMPIRICAL_LOG,

rule simulate_trials:
    output:
        f"{ROOT_OUTDIR}/trials/input_simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.tsv"
    params:
        root_outdir=ROOT_OUTDIR,
        scripts_dir=SCRIPTS_DIR,
        mlp=MLP,
    log:
        f"{ROOT_OUTDIR}/trials/simulate_trials-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        cd {params.root_outdir}
        time \
        sh {params.scripts_dir}/simulate.sh \
            {params.mlp} \
            trials \
            trials \
            {wildcards.year} \
            {wildcards.site} \
            {wildcards.treatment} \
            {wildcards.entry} \
            {wildcards.replication} \
            {wildcards.hidden_layer} > {log}
        """

rule simulate_gp:
    output:
        f"{ROOT_OUTDIR}/gp/input_simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.tsv"
    params:
        root_outdir=ROOT_OUTDIR,
        scripts_dir=SCRIPTS_DIR,
        mlp=MLP,
    log:
        f"{ROOT_OUTDIR}/gp/simulate_gp-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        cd {params.root_outdir}
        time \
        sh {params.scripts_dir}/simulate.sh \
            {params.mlp} \
            gp \
            {params.root_outdir}/gp \
            {wildcards.data_type} \
            {wildcards.n} \
            {wildcards.p} \
            {wildcards.hidden_layers} > {log}
        """

checkpoint empiricalprep_trials:
    input:
        f"{TRIALS_AGRIDAT_DIR}/{{fname}}"
    output:
        directory(f"{ROOT_OUTDIR}/trials/TMPDIR-{{fname}}")
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
    log:
        f"{ROOT_OUTDIR}/trials/TMPDIR-{{fname}}/log"
    conda:
        "conda.yaml"
    shell:
        """
        mkdir -p {output}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            trials \
            {input} \
            {output} > {log}
        """


# def get_outputs(wildcards):
#     ckpt = checkpoints.empiricalprep_trials.get(agridat_dir=wildcards.agridat_dir, fname=wildcards.fname)
#     output_dir = ckpt.output[0]
#     import os
#     return expand("{file}", file=os.listdir(output_dir))


# rule empiricalprep_trials:
#     output:
#     params:
#     log:
#     conda:
#         "conda.yaml"
#     shell:
#         """
#         cd {ROOT_OUTDIR}
#         time \
#         Rscript {params.scripts_dir}/empiricalprep.R \
#             trials \
#             ??? > {log}
#         """