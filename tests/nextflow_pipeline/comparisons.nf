process comparisons {
    publishDir "${params.root_outdir}/${analysis_type}", mode: 'copy'
    
    input:
        tuple val(analysis_type), path(linear_file), path(trees_file), path(mlp_file)
    
    output:
        path "output-${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_TREES-COMPARISON.tsv",
        path "output-${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_MLP-COMPARISON.tsv",
        path "output-${linear_file.baseName.replace('-LINEAR', '')}-TREES_vs_MLP-COMPARISON.tsv"
    
    script:
    """
    Rscript ${params.scripts_dir}/comparison.R \
        ${analysis_type} \
        ${linear_file} \
        ${trees_file} \
        .
    Rscript ${params.scripts_dir}/comparison.R \
        ${analysis_type} \
        ${linear_file} \
        ${mlp_file} \
        .
    Rscript ${params.scripts_dir}/comparison.R \
        ${analysis_type} \
        ${trees_file} \
        ${mlp_file} \
        .
    """
}
