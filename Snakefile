import os
from pathlib import Path

ROOT_OUTDIR = str(Path.home() / "Documents/mlp/tests/tmp")
SCRIPTS_DIR = str(Path.home() / "Documents/mlp/tests/scripts")
MLP = str(Path.home() / "Documents/mlp/target/release/mlp")
ANALYSIS_TYPES = ["trials", "gp"]
N_YEARS = [2]
N_SITES = [5]
N_TREATMENTS = [3]
N_ENTRIES = [10, 50]
N_REPLICATIONS = [3]
N_HIDDEN_LAYERS = [1, 2]
DATA_TYPES = ["CONTINUOUS", "BINARY"]
# N_OBSERVATIONS = [700]
# N_FEATURES = [42000]
N_OBSERVATIONS = [60]
N_FEATURES = [100]
TRIALS_AGRIDAT_DIR = str(Path.home() / "Documents/mlp/tests/datasets/agridat")
TRIALS_AGRIDAT_FNAMES = ["australia.soybean.txt", "ilri.sheep.txt"]
GP_AZODI2019_DIR = str(Path.home() / "Documents/mlp/tests/datasets/azodi_2019")
GP_AZODI2019_FNAMES = ["sorghum_geno.csv"]
EXCLUDE_LM = True
EXCLUDE_SOMMER = True
VERBOSE = True

TRIALS_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/trials/simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.tsv",
    year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
)

GP_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/gp/simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.tsv",
    data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
)

def TRIALS_EMPIRICAL_INPUT(wildcards):
    final_files = []
    for fname in TRIALS_AGRIDAT_FNAMES:
        manifest_path = checkpoints.empiricalprep_trials.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename:
                    final_files.append(f"{ROOT_OUTDIR}/trials/{filename}")
    return final_files

def GP_EMPIRICAL_INPUT(wildcards):
    final_files = []
    for fname in GP_AZODI2019_FNAMES:
        manifest_path = checkpoints.empiricalprep_gp.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename:
                    final_files.append(f"{ROOT_OUTDIR}/gp/{filename}")
    return final_files

def LINEAR_ANALYSIS_OUTPUT(wildcards):
    # TODO
    return []

def MLP_OUTPUT(wildcards):
    # TODO
    return []

rule all:
    input:
        TRIALS_SIMULATED_INPUT,
        GP_SIMULATED_INPUT,
        TRIALS_EMPIRICAL_INPUT,
        GP_EMPIRICAL_INPUT

rule simulate_trials:
    output:
        f"{ROOT_OUTDIR}/trials/simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.tsv"
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
        bash {params.scripts_dir}/simulate.sh \
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
        f"{ROOT_OUTDIR}/gp/simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.tsv"
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
        bash {params.scripts_dir}/simulate.sh \
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
        manifest=f"{ROOT_OUTDIR}/trials/{{fname}}.manifest.txt"
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
        tmpdir=temp(directory(f"{ROOT_OUTDIR}/trials/TMPDIR-{{fname}}"))
    log:
        f"{ROOT_OUTDIR}/trials/{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        mkdir -p {params.tmpdir}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            trials \
            {input} \
            {params.tmpdir} > {log}
        ls -1 {params.tmpdir} > {output.manifest}
        mv {params.tmpdir}/* {ROOT_OUTDIR}/trials/
        rm -rf {params.tmpdir}
        """

checkpoint empiricalprep_gp:
    input:
        f"{GP_AZODI2019_DIR}/{{fname}}"
    output:
        manifest=f"{ROOT_OUTDIR}/gp/{{fname}}.manifest.txt"
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
        tmpdir=temp(directory(f"{ROOT_OUTDIR}/gp/TMPDIR-{{fname}}"))
    log:
        f"{ROOT_OUTDIR}/gp/{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        mkdir -p {params.tmpdir}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            gp \
            {input} \
            {params.tmpdir} > {log}
        ls -1 {params.tmpdir} > {output.manifest}
        mv {params.tmpdir}/* {ROOT_OUTDIR}/gp/
        rm -rf {params.tmpdir}
        """

# TODO: probably checkpoints again for linear and mlp analyses

rule linear_analysis_trials:
    input:
        f"{ROOT_OUTDIR}/trials/{{fname}}.tsv"
    output:
        f"{ROOT_OUTDIR}/trials/output-{{fname}}-LINEAR.tsv"
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
        exclude_lm = EXCLUDE_LM,
        exclude_sommer = EXCLUDE_SOMMER,
        verbose = VERBOSE
    log:
        f"{ROOT_OUTDIR}/trials/linear_analysis-{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        IS_SIMULATED = if [[ $(basename {input} | grep -c "^simulated") -gt 0 ]]; then echo "TRUE"; else echo "FALSE"; fi;
        time \
        Rscript {params.scripts_dir}/linear.R \
            trials \
            {input} \
            {output} \
            $IS_SIMULATED \
            {params.exclude_lm} \
            {params.exclude_sommer} \
            {params.verbose} > {log}
        """