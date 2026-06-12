// Randomisation for GP CV and remote-sensing analysis for use across linear, trees, and mlp models (GPU is not required)

process randomisation_gp_rs {
    // publishDir "${params.root_outdir}/${analysis_type}", mode: 'copy'
    publishDir "${params.root_outdir}", 
        mode: 'copy', 
        saveAs: { filename -> "${analysis_type}/${filename}" }
    
    input:
        tuple val(analysis_type), path(data_file)
    
    output:
        tuple val(analysis_type), path(data_file), path("output-${data_file.baseName}-RANDOMISATION.tsv")
    
    script:
    """
    bash ${params.scripts_dir}/randomisationgprs.sh \
        ${analysis_type} \
        ${data_file} \
        . \
        ${params.n_reps} \
        ${params.n_folds} \
        ${params.seed}
    # mv output-*-RANDOMISATION.tsv output-${data_file.baseName}-RANDOMISATION.tsv
    """
}
