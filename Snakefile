import os
from pathlib import Path

ROOT_OUTDIR = str(Path.home() / "Documents/mlp/tests/tmp")
SCRIPTS_DIR = str(Path.home() / "Documents/mlp/tests/scripts")
MLP = str(Path.home() / "Documents/mlp/target/release/mlp")
ANALYSIS_TYPES = ["trials", "gp"]
wildcard_constraints:
    analysis_type="trials|gp"
N_YEARS = [2]
N_SITES = [5]
N_TREATMENTS = [3]
N_ENTRIES = [10, 50]
N_REPLICATIONS = [3]
N_HIDDEN_LAYERS = [1, 2]
DATA_TYPES = ["CONTINUOUS", "BINARY"]
N_OBSERVATIONS = [500]
N_FEATURES = [1000]
TRIALS_AGRIDAT_DIR = str(Path.home() / "Documents/mlp/tests/datasets/agridat")
TRIALS_AGRIDAT_FNAMES = ["australia.soybean.txt", "ilri.sheep.txt"]
GP_AZODI2019_DIR = str(Path.home() / "Documents/mlp/tests/datasets/azodi_2019")
GP_AZODI2019_FNAMES = ["rice_geno.csv", "sorghum_geno.csv", "spruce_geno.csv"]
EXCLUDE_LM = "FALSE"
EXCLUDE_SOMMER = "TRUE"
N_FOLDS = 5
N_REPS = 3
N_ITERATIONS = 100
N_BURNIN_ITERATIONS = 10
MODELS = "BRR,BayesA"
BASE_SEED = 42
VERBOSE = "TRUE"

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
    final_files = []
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/trials/output-simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}-LINEAR.tsv",
        year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/gp/output-simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}-LINEAR.tsv",
        data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    for fname in TRIALS_AGRIDAT_FNAMES:
        manifest_path = checkpoints.empiricalprep_trials.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/trials/output-{dataset_base}-LINEAR.tsv")
    for fname in GP_AZODI2019_FNAMES:
        manifest_path = checkpoints.empiricalprep_gp.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/gp/output-{dataset_base}-LINEAR.tsv")
    return final_files

def MLP_ANALYSIS_OUTPUT(wildcards):
    final_files = []
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/trials/output-simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}-MLP.tsv",
        year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/gp/output-simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}-MLP.tsv",
        data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    for fname in TRIALS_AGRIDAT_FNAMES:
        manifest_path = checkpoints.empiricalprep_trials.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/trials/output-{dataset_base}-MLP.tsv")
    for fname in GP_AZODI2019_FNAMES:
        manifest_path = checkpoints.empiricalprep_gp.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/gp/output-{dataset_base}-MLP.tsv")
    return final_files

def COMPARISONS_OUTPUT(wildcards):
    fnames_tmp = LINEAR_ANALYSIS_OUTPUT(wildcards) + MLP_ANALYSIS_OUTPUT(wildcards)
    final_files = []
    for fname in fnames_tmp:
        fname = fname.replace("-LINEAR.tsv", "-COMPARISON.tsv")
        fname = fname.replace("-MLP.tsv", "-COMPARISON.tsv")
        final_files.append(fname)
    return final_files

rule all:
    input:
        TRIALS_SIMULATED_INPUT,
        GP_SIMULATED_INPUT,
        TRIALS_EMPIRICAL_INPUT,
        GP_EMPIRICAL_INPUT,
        LINEAR_ANALYSIS_OUTPUT,
        MLP_ANALYSIS_OUTPUT,
        COMPARISONS_OUTPUT

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

rule linear_analysis:
    input:
        f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR.tsv",
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        exclude_lm = EXCLUDE_LM,
        exclude_sommer = EXCLUDE_SOMMER,
        n_folds = N_FOLDS,
        n_reps = N_REPS,
        n_iterations = N_ITERATIONS,
        n_burnin_iterations = N_BURNIN_ITERATIONS,
        models = MODELS,
        base_seed = BASE_SEED,
        verbose = VERBOSE
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/linear_analysis-{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        if [[ {wildcards.analysis_type} == "trials" ]]; then
            if [[ $(basename {input} | grep -c "^simulated") -gt 0 ]]; then 
                IS_SIMULATED="TRUE"
            else 
                IS_SIMULATED="FALSE"
            fi;
            time \
            Rscript {params.scripts_dir}/linear.R \
                trials \
                {input} \
                {params.outdir} \
                $IS_SIMULATED \
                {params.exclude_lm} \
                {params.exclude_sommer} \
                {params.verbose} > {log}
        else
            time \
            Rscript {params.scripts_dir}/linear.R \
                gp \
                {input} \
                {params.outdir} \
                {params.n_reps} \
                {params.n_folds} \
                {params.n_iterations} \
                {params.n_burnin_iterations} \
                {params.models} \
                {params.base_seed} \
                {params.verbose} > {log}
        fi;
        """

rule mlp_analysis:
    input:
        f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-MLP.tsv",
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        n_folds = N_FOLDS,
        n_reps = N_REPS,
        n_iterations = N_ITERATIONS,
        n_burnin_iterations = N_BURNIN_ITERATIONS,
        base_seed = BASE_SEED
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/mlp_analysis-{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        if [[ {wildcards.analysis_type} == "trials" ]]; then
            time \
            bash {params.scripts_dir}/mlp.sh \
                {params.mlp} \
                trials \
                {input} \
                {params.outdir} > {log}
        else
            time \
            bash {params.scripts_dir}/mlp.sh \
                {params.mlp} \
                gp \
                {input} \
                {params.outdir} \
                {params.n_reps} \
                {params.n_folds} \
                {params.base_seed} > {log}
        fi;
        """

rule comparisons:
    input:
        linear=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR.tsv",
        mlp=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-MLP.tsv"
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-COMPARISON.tsv"
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}"
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/comparison-{{fname}}.log"
    conda:
        "conda.yaml"
    shell:
        """
        time \
        Rscript {params.scripts_dir}/comparison.R \
            {wildcards.analysis_type} \
            {input.linear} \
            {input.mlp} \
            {params.outdir} > {log}
        """