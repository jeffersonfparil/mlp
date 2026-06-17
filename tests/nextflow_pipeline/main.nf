#!/usr/bin/env nextflow

include { simulate_trials; simulate_gp } from './simulations.nf'
include { empiricalprep_trials; empiricalprep_gp; empiricalprep_remotesensing } from './empiricalprep.nf'
include { randomisation_gp_rs } from './randomisation.nf'
include { linear_analysis } from './model_linear.nf'
include { trees_analysis } from './model_trees.nf'
include { mlp_analysis } from './model_mlp.nf'
include { comparisons } from './comparisons.nf'

def generate_trials_params() {
    params.n_years.collect { y ->
        params.n_sites.collect { s ->
            params.n_treatments.collect { t ->
                params.n_entries.collect { e ->
                    params.n_replications.collect { r ->
                        params.n_hidden_layers.collect { h ->
                            tuple(y, s, t, e, r, h)
                        }
                    }
                }
            }
        }
    }.flatten().collate(6)
}

def generate_gp_params() {
    params.data_types.collect { d ->
        params.n_observations.collect { n ->
            params.n_features.collect { p ->
                params.n_hidden_layers.collect { h ->
                    tuple(d, n, p, h)
                }
            }
        }
    }.flatten().collate(4)
}

def get_analysis_type(filename) {
    filename.contains('simulated-YEARS') || filename.contains('australia.soybean') ? 'trials' : 'gp'
}

workflow {
    trials_simulated = Channel.fromList(generate_trials_params())
        | simulate_trials
        | flatten()
        | map { file -> tuple('trials', file) }
    
    trials_empirical = Channel.fromList(files("${params.trials_agridat_dir}/*.txt"))
        | empiricalprep_trials
        | flatten()
        | map { file -> tuple('trials', file) }

    gp_simulated = Channel.fromList(generate_gp_params())
        | simulate_gp
        | flatten()
        | map { file -> tuple('gp', file) }

    gp_empirical = Channel.fromList(files("${params.gp_azodi2019_dir}/*_geno.csv"))
        | empiricalprep_gp
        | flatten()
        | map { file -> tuple('gp', file) }

    remotesensing_empirical = Channel.fromList(params.remotesensing_farag_2024_dates)
        | empiricalprep_remotesensing
        | flatten()
        | map { file -> tuple('remotesensing', file) }

    combined = trials_simulated
        | mix(trials_empirical)
        | mix(gp_simulated)
        | mix(gp_empirical)
        | mix(remotesensing_empirical)

    randomisations = randomisation_gp_rs(combined)
    
    linear_output = randomisations
        | linear_analysis
        // | view()

    trees_output = randomisations
        | trees_analysis
        // | view()
    
    mlp_output = randomisations
        | mlp_analysis
        // | view()

    all_outputs = linear_output
        | mix(trees_output)
        | mix(mlp_output)
        | map {
            analysis, file -> 
            // def dir = file.parent
            def id = file.baseName.split('-')[1..-2].join('-')
            def linear = "${params.root_outdir}/${analysis}/output-${id}-LINEAR.tsv"
            def trees = "${params.root_outdir}/${analysis}/output-${id}-TREES.tsv"
            def mlp = "${params.root_outdir}/${analysis}/output-${id}-MLP.tsv"
            return tuple(analysis, linear, trees, mlp)
        }
        | unique()
        | toList()
        | flatMap {it}
        // | view()
    
    all_outputs
        | comparisons
        // | view()
}
