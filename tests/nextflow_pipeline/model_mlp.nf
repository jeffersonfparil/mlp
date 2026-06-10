process mlp_analysis {
    publishDir "${params.root_outdir}/${analysis_type}", mode: 'copy'
    
    input:
        tuple val(analysis_type), path(data_file), path(randomisation_file)
    
    output:
        tuple val(analysis_type), path("output-${data_file.baseName}-MLP.tsv")
    
    script:
    def is_trials = analysis_type == "trials"
    
    if (is_trials) {
        """
        bash ${params.scripts_dir}/mlp.sh \
            ${params.mlp} \
            ${analysis_type} \
            ${data_file} \
            .
        """
    } else {
        """
        bash ${params.scripts_dir}/mlp.sh \
            ${params.mlp} \
            ${analysis_type} \
            ${data_file} \
            . \
            ${randomisation_file} \
            ${params.n_reps} \
            ${params.n_folds}
        """
    }
}
