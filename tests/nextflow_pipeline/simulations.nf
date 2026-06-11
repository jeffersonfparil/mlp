// Simulate datasets for trial analysis and GP CV using mlp (GPU required)

process simulate_trials {
    label 'gpu'

    publishDir "${params.root_outdir}/trials", mode: 'copy'
    
    input:
        tuple val(year), val(site), val(treatment), val(entry), val(replication), val(hidden_layer)
    
    output:
        path "simulated-*.tsv"
    
    script:
    """
    bash ${params.scripts_dir}/simulate.sh \
        ${params.mlp} \
        trials \
        . \
        ${year} \
        ${site} \
        ${treatment} \
        ${entry} \
        ${replication} \
        ${hidden_layer}
    """
}

process simulate_gp {
    label 'gpu'

    publishDir "${params.root_outdir}/gp", mode: 'copy'
    
    input:
        tuple val(data_type), val(n), val(p), val(hidden_layers)
    
    output:
        path "simulated-*.tsv"
    
    script:
    """
    bash ${params.scripts_dir}/simulate.sh \
        ${params.mlp} \
        gp \
        . \
        ${data_type} \
        ${n} \
        ${p} \
        ${hidden_layers}
    """
}
