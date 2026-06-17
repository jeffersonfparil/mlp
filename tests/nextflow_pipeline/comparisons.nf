process comparisons {
    publishDir "${params.root_outdir}", 
        mode: 'copy', 
        saveAs: { filename -> "${analysis_type}/${filename}" }
    
    input:
        tuple val(analysis_type), path(linear_file), path(trees_file), path(mlp_file)
    
    output:
        tuple path("${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_TREES-COMPARISON.tsv"),
              path("${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_MLP-COMPARISON.tsv"),
              path("${linear_file.baseName.replace('-LINEAR', '')}-TREES_vs_MLP-COMPARISON.tsv"),
              path("${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_TREES-COMPARISON.png"),
              path("${linear_file.baseName.replace('-LINEAR', '')}-LINEAR_vs_MLP-COMPARISON.png"),
              path("${linear_file.baseName.replace('-LINEAR', '')}-TREES_vs_MLP-COMPARISON.png")
    
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
