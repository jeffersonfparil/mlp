// Processing of empirical data in preparation for trial analysis, GP CV and remote-sensing analysis (GPU is not required)

process empiricalprep_trials {
    publishDir "${params.root_outdir}/trials", mode: 'copy'
    
    input:
        val fname
    
    output:
        path "*.tsv"
    
    script:
    """
    mkdir -p ${fname}.tmpdir
    Rscript ${params.scripts_dir}/empiricalprep.R \
        trials \
        ${fname} \
        ${fname}.tmpdir
    mv ${fname}.tmpdir/* .
    rm -rf ${fname}.tmpdir
    """
}

process empiricalprep_gp {
    publishDir "${params.root_outdir}/gp", mode: 'copy'
    
    input:
        val fname
    
    output:
        path "*.tsv"
    
    script:
    """
    mkdir -p ${fname}.tmpdir
    Rscript ${params.scripts_dir}/empiricalprep.R \
        gp \
        ${fname} \
        ${fname}.tmpdir
    mv ${fname}.tmpdir/* .
    rm -rf ${fname}.tmpdir
    """
}

process empiricalprep_remotesensing {
    publishDir "${params.root_outdir}/remotesensing", mode: 'copy'
    
    input:
        val date
    
    output:
        path "*.tsv"
    
    script:
    """
    pixi run --manifest-path ${params.pixi_toml} \
    Rscript ${params.scripts_dir}/empiricalprep.R \
        remotesensing \
        ${params.remotesensing_farag_2024_fname_trait_csv} \
        "," \
        ${params.remotesensing_farag_2024_trait} \
        ${params.remotesensing_farag_2024_dir} \
        ${date} \
        ./
    """
}

