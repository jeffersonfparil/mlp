process trees_analysis {
    publishDir "${params.root_outdir}/${analysis_type}", mode: 'copy'
    
    input:
        tuple val(analysis_type), path(data_file), path(randomisation_file)
    
    output:
        path "output-${data_file.baseName}-TREES.tsv"
    
    script:
    def is_trials = analysis_type == "trials"
    
    if (is_trials) {
        """
        python ${params.scripts_dir}/trees.py \
            ${analysis_type} \
            ${data_file} \
            . \
            --n-estimators=${params.n_estimators_trees} \
            --max-depth=${params.max_depth_trees}
        """
    } else {
        """
        python ${params.scripts_dir}/trees.py \
            ${analysis_type} \
            ${data_file} \
            . \
            --randomisation-input-file=${randomisation_file} \
            --n-replicates=${params.n_reps} \
            --n-folds=${params.n_folds}
        """
    }
}
