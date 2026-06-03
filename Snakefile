import os
from pathlib import Path

ROOT_OUTDIR = str(Path.home() / "Documents/mlp/tests/tmp")
SCRIPTS_DIR = str(Path.home() / "Documents/mlp/tests/scripts")
MLP = str(Path.home() / "Documents/mlp/target/release/mlp")
ANALYSIS_TYPES = ["trials", "gp"]
N_YEARS = [2]
N_SITES = [5]
N_TREATMENTS = [3]
N_ENTRIES = [10]
N_REPLICATIONS = [3]
N_HIDDEN_LAYERS = [1]
DATA_TYPES = ["CONTINUOUS"]
N_OBSERVATIONS = [500]
N_FEATURES = [1000]
TRIALS_AGRIDAT_DIR = str(Path.home() / "Documents/mlp/tests/datasets/agridat")
# TRIALS_AGRIDAT_FNAMES = ["australia.soybean.txt", "ilri.sheep.txt"]
TRIALS_AGRIDAT_FNAMES = ["australia.soybean.txt"]
GP_AZODI2019_DIR = str(Path.home() / "Documents/mlp/tests/datasets/azodi_2019")
# GP_AZODI2019_FNAMES = ["sorghum_geno.csv", "rice_geno.csv", "spruce_geno.csv"]
# GP_AZODI2019_FNAMES = ["sorghum_geno.csv"]
GP_AZODI2019_FNAMES = ["test_geno.csv"]
REMOTESENSING_FARAG_2024_DIR = str(Path.home() / "Documents/mlp/tests/datasets/farag_2024")
REMOTESENSING_FARAG_2024_FNAME_TRAIT_CSV = REMOTESENSING_FARAG_2024_DIR + "/constant_agronomic_traits_2021.csv"
REMOTESENSING_FARAG_2024_TRAIT = "Yield"
REMOTESENSING_FARAG_2024_DATES = ["06142021", "07142021", "08032021", "09032021"]
EXCLUDE_LM = "FALSE"
EXCLUDE_LMER = "TRUE"
EXCLUDE_SOMMER = "TRUE"
EXCLUDE_ASREML = "TRUE"
N_FOLDS = 2
N_REPS = 2
N_ITERATIONS_LINEAR = 100
N_BURNIN_ITERATIONS_LINEAR = 100
N_EPOCHS_MLP = 1000
N_BURNIN_EPOCHS_MLP = 100
MODELS = "BayesA"
BASE_SEED = 42
VERBOSE = "TRUE"
wildcard_constraints:
    analysis_type="trials|gp|remotesensing",
    date="|".join(REMOTESENSING_FARAG_2024_DATES)


TRIALS_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/trials/simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}.tsv",
    year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
)

GP_SIMULATED_INPUT = expand(
    f"{ROOT_OUTDIR}/gp/simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}.tsv",
    data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
)

REMOTESENSING_EMPIRICAL_INPUT = expand(
    f"{ROOT_OUTDIR}/remotesensing/{REMOTESENSING_FARAG_2024_TRAIT}_{{date}}.tsv",
    date=REMOTESENSING_FARAG_2024_DATES
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

def RANDOMISATION_GP_OUTPUT(wildcards):
    final_files = []
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/gp/output-simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}-RANDOMISATION.tsv",
        data_type=DATA_TYPES, n=N_OBSERVATIONS, p=N_FEATURES, hidden_layers=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    for fname in GP_AZODI2019_FNAMES:
        manifest_path = checkpoints.empiricalprep_gp.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/gp/output-{dataset_base}-RANDOMISATION.tsv")
    for date in REMOTESENSING_FARAG_2024_DATES:
        final_files.append(f"{ROOT_OUTDIR}/remotesensing/output-{REMOTESENSING_FARAG_2024_TRAIT}_{date}-RANDOMISATION.tsv")
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
    for date in REMOTESENSING_FARAG_2024_DATES:
        final_files.append(f"{ROOT_OUTDIR}/remotesensing/output-{REMOTESENSING_FARAG_2024_TRAIT}_{date}-LINEAR.tsv")
    return final_files

def TREES_ANALYSIS_OUTPUT(wildcards):
    final_files = []
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/trials/output-simulated-YEARS_{{year}}-SITES_{{site}}-TREATMENTS_{{treatment}}-ENTRIES_{{entry}}-REPLICATIONS_{{replication}}-HIDDEN_LAYERS_{{hidden_layer}}-TREES.tsv",
        year=N_YEARS, site=N_SITES, treatment=N_TREATMENTS, entry=N_ENTRIES, replication=N_REPLICATIONS, hidden_layer=N_HIDDEN_LAYERS
    )
    final_files.extend(simulated_targets)
    simulated_targets = expand(
        f"{ROOT_OUTDIR}/gp/output-simulated-DATA_TYPE_{{data_type}}-N_{{n}}-P_{{p}}-HIDDEN_LAYERS_{{hidden_layers}}-TREES.tsv",
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
                    final_files.append(f"{ROOT_OUTDIR}/trials/output-{dataset_base}-TREES.tsv")
    for fname in GP_AZODI2019_FNAMES:
        manifest_path = checkpoints.empiricalprep_gp.get(fname=fname).output.manifest
        with open(manifest_path, "r") as f:
            for line in f:
                filename = line.strip()
                if filename.endswith(".tsv"):
                    dataset_base = filename.replace(".tsv", "")
                    final_files.append(f"{ROOT_OUTDIR}/gp/output-{dataset_base}-TREES.tsv")
    for date in REMOTESENSING_FARAG_2024_DATES:
        final_files.append(f"{ROOT_OUTDIR}/remotesensing/output-{REMOTESENSING_FARAG_2024_TRAIT}_{date}-TREES.tsv")
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
    for date in REMOTESENSING_FARAG_2024_DATES:
        final_files.append(f"{ROOT_OUTDIR}/remotesensing/output-{REMOTESENSING_FARAG_2024_TRAIT}_{date}-MLP.tsv")
    return final_files

def COMPARISONS_OUTPUT(wildcards):
    fnames_tmp = LINEAR_ANALYSIS_OUTPUT(wildcards) + MLP_ANALYSIS_OUTPUT(wildcards)
    final_files = []
    for fname in fnames_tmp:
        fname_0 = fname.replace("-LINEAR.tsv", "-LINEAR_vs_TREES-COMPARISON.tsv")
        fname_1 = fname.replace("-LINEAR.tsv", "-LINEAR_vs_MLP-COMPARISON.tsv")
        fname_2 = fname.replace("-LINEAR.tsv", "-TREES_vs_MLP-COMPARISON.tsv")
        final_files.append(fname_0)
        final_files.append(fname_1)
        final_files.append(fname_2)
    return final_files

rule all:
    input:
        TRIALS_SIMULATED_INPUT,
        GP_SIMULATED_INPUT,
        TRIALS_EMPIRICAL_INPUT,
        GP_EMPIRICAL_INPUT,
        REMOTESENSING_EMPIRICAL_INPUT,
        RANDOMISATION_GP_OUTPUT,
        LINEAR_ANALYSIS_OUTPUT,
        TREES_ANALYSIS_OUTPUT,
        # MLP_ANALYSIS_OUTPUT,
        # COMPARISONS_OUTPUT,

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
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="gpu",
        slurm_extra="'--gres=gpu:h100:1'",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
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
            {wildcards.hidden_layer} > {log} 2>&1
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
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="gpu",
        slurm_extra="'--gres=gpu:h100:1'",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
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
            {wildcards.hidden_layers} > {log} 2>&1
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
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        mkdir -p {params.tmpdir}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            trials \
            {input} \
            {params.tmpdir} > {log} 2>&1
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
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        mkdir -p {params.tmpdir}
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            gp \
            {input} \
            {params.tmpdir} > {log} 2>&1
        ls -1 {params.tmpdir} > {output.manifest}
        mv {params.tmpdir}/* {ROOT_OUTDIR}/gp/
        rm -rf {params.tmpdir}
        """

rule empiricalprep_remotesensing:
    input:
        f"{REMOTESENSING_FARAG_2024_DIR}/{{date}}"
    output:
        f"{ROOT_OUTDIR}/remotesensing/{REMOTESENSING_FARAG_2024_TRAIT}_{{date}}.tsv"
    params:
        scripts_dir=SCRIPTS_DIR,
        fname_trait=REMOTESENSING_FARAG_2024_FNAME_TRAIT_CSV,
        fname_trait_delim=",",
        target=REMOTESENSING_FARAG_2024_TRAIT,
        image_root_dir=REMOTESENSING_FARAG_2024_DIR,
        dirname_output=f"{ROOT_OUTDIR}/remotesensing/"
    log:
        f"{ROOT_OUTDIR}/remotesensing/{REMOTESENSING_FARAG_2024_TRAIT}_{{date}}.log"
    conda:
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        time \
        Rscript {params.scripts_dir}/empiricalprep.R \
            remotesensing \
            {params.fname_trait} \
            {params.fname_trait_delim} \
            {params.target} \
            {params.image_root_dir} \
            {wildcards.date} \
            {params.dirname_output} > {log} 2>&1
        """

rule randomisation_gp:
    input:
        f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-RANDOMISATION.tsv",
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        n_folds = N_FOLDS,
        n_reps = N_REPS,
        base_seed = BASE_SEED
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/randomisation-{{fname}}.log"
    conda:
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        time \
        bash {params.scripts_dir}/randomisationgp.sh \
            {input} \
            {params.outdir} \
            {params.n_reps} \
            {params.n_folds} \
            {params.base_seed} > {log} 2>&1
        """

rule linear_analysis:
    input:
        data=f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
        randomisation=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-RANDOMISATION.tsv",
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR.tsv",
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        exclude_lm = EXCLUDE_LM,
        exclude_lmer = EXCLUDE_LMER,
        exclude_sommer = EXCLUDE_SOMMER,
        exclude_asreml = EXCLUDE_ASREML,
        n_folds = N_FOLDS,
        n_reps = N_REPS,
        n_iterations = N_ITERATIONS_LINEAR,
        n_burnin_iterations = N_BURNIN_ITERATIONS_LINEAR,
        models = MODELS,
        base_seed = BASE_SEED,
        verbose = VERBOSE
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/linear_analysis-{{fname}}.log"
    conda:
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        if [[ {wildcards.analysis_type} == "trials" ]]; then
            if [[ $(basename {input} | grep -c "^simulated") -gt 0 ]]; then 
                IS_SIMULATED="TRUE"
            else 
                IS_SIMULATED="FALSE"
            fi;
            if [[ {params.exclude_asreml} == "FALSE" ]]; then
                module try-load ASReml-R > {log} 2>&1
            fi;
            time \
            Rscript {params.scripts_dir}/linear.R \
                trials \
                {input.data} \
                {params.outdir} \
                $IS_SIMULATED \
                {params.exclude_lm} \
                {params.exclude_lmer} \
                {params.exclude_sommer} \
                {params.exclude_asreml} \
                {params.verbose} >> {log} 2>&1
        else
            time \
            Rscript {params.scripts_dir}/linear.R \
                gp \
                {input.data} \
                {params.outdir} \
                {input.randomisation} \
                {params.n_reps} \
                {params.n_folds} \
                {params.n_iterations} \
                {params.n_burnin_iterations} \
                {params.models} \
                {params.base_seed} \
                {params.verbose} > {log} 2>&1
        fi;
        """

rule trees_analysis:
    input:
        data=f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
        randomisation=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-RANDOMISATION.tsv", # not needed for trials and only here for gp to ensure that the gp randomisation step is run before this step, as it is needed for the linear analysis and we want to ensure that the linear and trees analyses are run in the same order for gp
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-TREES.tsv",
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        n_reps = N_REPS,
        n_folds = N_FOLDS,
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/trees_analysis-{{fname}}.log"
    conda:
        "trees.yaml"
    threads: 1
    resources:
        slurm_partition="gpu",
        slurm_extra="'--gres=gpu:h100:1'",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        if [[ {wildcards.analysis_type} == "trials" ]]; then
            time \
            python {params.scripts_dir}/trees.py \
                trials \
                {input.data} \
                {params.outdir} \
                "." \
                {params.n_reps} \
                {params.n_folds} >> {log} 2>&1
        else
            time \
            python {params.scripts_dir}/trees.py \
                gp \
                {input.data} \
                {params.outdir} \
                {input.randomisation} \
                {params.n_reps} \
                {params.n_folds} >> {log} 2>&1
        fi;
        """

rule mlp_analysis:
    input:
        data=f"{ROOT_OUTDIR}/{{analysis_type}}/{{fname}}.tsv",
        randomisation=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-RANDOMISATION.tsv",
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-MLP.tsv",
    params:
        mlp=MLP,
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}",
        n_folds = N_FOLDS,
        n_reps = N_REPS,
        n_epochs = N_EPOCHS_MLP,
        n_burnin_epochs = N_BURNIN_EPOCHS_MLP,
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/mlp_analysis-{{fname}}.log"
    conda:
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="gpu",
        slurm_extra="'--gres=gpu:h100:1'",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        if [[ {wildcards.analysis_type} == "trials" ]]; then
            time \
            bash {params.scripts_dir}/mlp.sh \
                {params.mlp} \
                trials \
                {input.data} \
                {params.outdir} > {log} 2>&1
        else
            time \
            bash {params.scripts_dir}/mlp.sh \
                {params.mlp} \
                gp \
                {input.data} \
                {params.outdir} \
                {input.randomisation} \
                {params.n_reps} \
                {params.n_folds} \
                {params.n_epochs} \
                {params.n_burnin_epochs} > {log} 2>&1
        fi;
        """

rule comparisons:
    input:
        linear=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR.tsv",
        trees=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-TREES.tsv",
        mlp=f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-MLP.tsv"
    output:
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR_vs_TREES-COMPARISON.tsv",
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-LINEAR_vs_MLP-COMPARISON.tsv",
        f"{ROOT_OUTDIR}/{{analysis_type}}/output-{{fname}}-TREES_vs_MLP-COMPARISON.tsv",
    params:
        scripts_dir=SCRIPTS_DIR,
        outdir=f"{ROOT_OUTDIR}/{{analysis_type}}"
    log:
        f"{ROOT_OUTDIR}/{{analysis_type}}/comparison-{{fname}}.log"
    conda:
        "general.yaml"
    threads: 1
    resources:
        slurm_partition="cpu",
        tasks=1,
        nodes=1,
        mem_mb=1064,
        runtime=1440,
    shell:
        """
        echo "LINEAR vs TREES comparison:" > {log} 2>&1
        time \
        Rscript {params.scripts_dir}/comparison.R \
            {wildcards.analysis_type} \
            {input.linear} \
            {input.trees} \
            {params.outdir} >> {log} 2>&1
        echo "LINEAR vs MLP comparison:" >> {log} 2>&1
        time \
        Rscript {params.scripts_dir}/comparison.R \
            {wildcards.analysis_type} \
            {input.linear} \
            {input.mlp} \
            {params.outdir} >> {log} 2>&1
        echo "TREES vs MLP comparison:" >> {log} 2>&1
        time \
        Rscript {params.scripts_dir}/comparison.R \
            {wildcards.analysis_type} \
            {input.trees} \
            {input.mlp} \
            {params.outdir} >> {log} 2>&1
        """