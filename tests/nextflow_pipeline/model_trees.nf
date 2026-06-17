process trees_analysis {
    label 'gpu'

    publishDir "${params.root_outdir}", 
        mode: 'copy', 
        saveAs: { filename -> "${analysis_type}/${filename}" }
    
    input:
        tuple val(analysis_type), path(data_file), path(randomisation_file)
    
    output:
        tuple val(analysis_type), path("output-${data_file.baseName}-TREES.tsv")
    
    script:
    def is_trials = analysis_type == "trials"
    
    if (is_trials) {
        """
        python ${params.scripts_dir}/trees.py \
            ${analysis_type} \
            ${data_file} \
            . \
            --n-estimators=${params.n_estimators_trees} \
            --max-depth=${params.max_depth_trees} \
            --learning-rate=${params.learning_rate_trees} \
            --seed=${params.seed}
        """
    } else {
        """
        python ${params.scripts_dir}/trees.py \
            ${analysis_type} \
            ${data_file} \
            . \
            --randomisation-input-file=${randomisation_file} \
            --n-replicates=${params.n_reps} \
            --n-folds=${params.n_folds} \
            --early-stopping-rounds=${params.early_stopping_rounds_trees} \
            --optim-n-estimators=${params.optim_n_estimators_trees} \
            --optim-max-depth=${params.optim_max_depth_trees} \
            --optim-learning-rate=${params.optim_learning_rate_trees} \
            --optim-subsample=${params.optim_subsample_trees} \
            --seed=${params.seed}
        """
    }
}
