process linear_analysis {
    // publishDir "${params.root_outdir}/${analysis_type}", mode: 'copy'
    publishDir "${params.root_outdir}", 
        mode: 'copy', 
        saveAs: { filename -> "${analysis_type}/${filename}" }
    
    input:
        tuple val(analysis_type), path(data_file), path(randomisation_file)
    
    output:
        tuple val(analysis_type), path("output-${data_file.baseName}-LINEAR.tsv")
    
    script:
    def is_trials = analysis_type == "trials"

    if (is_trials) {
        """
        if [[ ${params.exclude_asreml} == "FALSE" ]]; then
            module try-load ASReml-R
        fi
        Rscript ${params.scripts_dir}/linear.R \
            ${analysis_type} \
            ${data_file} \
            . \
            ${params.exclude_lm} \
            ${params.exclude_lmer} \
            ${params.exclude_sommer} \
            ${params.exclude_asreml} \
            ${params.verbose}
        """
    } else {
        """
        Rscript ${params.scripts_dir}/linear.R \
            ${analysis_type} \
            ${data_file} \
            . \
            ${randomisation_file} \
            ${params.n_reps} \
            ${params.n_folds} \
            ${params.n_iterations_linear} \
            ${params.n_burnin_iterations_linear} \
            ${params.models_linear} \
            ${params.seed} \
            ${params.verbose}
        """
    }
}
