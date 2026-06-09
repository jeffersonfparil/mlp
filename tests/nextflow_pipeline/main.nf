#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { simulate_trials; simulate_gp } from './simulations.nf'
include { empiricalprep_trials; empiricalprep_gp; empiricalprep_remotesensing } from './empiricalprep.nf'
include { randomisation_gp_rs } from './randomisation.nf'
include { linear_analysis } from './model_linear.nf'
include { trees_analysis } from './model_trees.nf'
include { mlp_analysis } from './model_mlp.nf'
// include { comparisons } from './comparisons.nf'

// Helper function: generate parameter combinations
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
    
    // trials_params = Channel.fromList(generate_trials_params())
    // trials_simulated = simulate_trials(trials_params).flatten()
    

    
    // Trial simulations
    trials_params = Channel.fromList(generate_trials_params())
    trials_simulated = simulate_trials(trials_params).flatten()
    
    // GP simulations
    gp_params = Channel.fromList(generate_gp_params())
    gp_simulated = simulate_gp(gp_params).flatten()
    
    // Trial empirical data
    trials_fnames = Channel.fromList(params.trials_agridat_fnames)
    trials_empirical = empiricalprep_trials(trials_fnames).flatten()
    // trials_empirical_prep = empiricalprep_trials(trials_fnames)
    // trials_data = trials_empirical_prep.data.flatten()
    
    // GP empirical data
    gp_fnames = Channel.fromList(params.gp_azodi2019_fnames)
    gp_empirical = empiricalprep_gp(gp_fnames).flatten()
    // gp_empirical_prep = empiricalprep_gp(gp_fnames)
    // gp_data = gp_empirical_prep.data.flatten()
    
    // Remote sensing empirical data
    remotesensing_dates = Channel.fromList(params.remotesensing_farag_2024_dates)
    remotesensing_empirical = empiricalprep_remotesensing(remotesensing_dates).flatten()
    
    // Names
    input_tuple_analysis_input = trials_simulated.mix(trials_empirical).map { file -> tuple('trials', file) }.
        mix(gp_simulated.mix(gp_empirical).map { file -> tuple('gp', file) }).
        mix(remotesensing_empirical.map { file -> tuple('remotesensing', file) })
        
    // println "Input for randomisation:"
    // input_tuple_analysis_input.view()

    input_tuple_analysis_input_randomisation = randomisation_gp_rs(input_tuple_analysis_input)

    // println "Output from randomisation:"
    // input_tuple_analysis_input_randomisation.view()

    linear_results = linear_analysis(input_tuple_analysis_input_randomisation)
    println "Linear analysis results:"
    linear_results.view()




    


    // // Randomisation for GP only (simulated + empirical)
    // all_gp_data = gp_simulated.mix(gp_data)
    //     .map { file -> tuple('gp', file) }
    
    // randomisation_results = randomisation_gp(all_gp_data)
    
    // // Create data files with their randomisation results
    // rand_keyed = randomisation_results.map { file -> 
    //     def base = file.baseName.replace('output-', '').replace('-RANDOMISATION', '')
    //     tuple(base, file)
    // }
    
    // gp_with_rand = all_gp_data
    //     .map { analysis_type, data_file ->
    //         tuple(data_file.baseName, analysis_type, data_file)
    //     }
    //     .join(rand_keyed)
    //     .map { base, analysis_type, data_file, rand_file ->
    //         tuple(analysis_type, data_file, rand_file)
    //     }
    
    // // Trials data (use data file as placeholder for randomisation - not used)
    // trials_with_rand = trials_simulated.mix(trials_data)
    //     .map { file -> tuple('trials', file, file) }
    
    // // == ANALYSIS STAGE ==
    
    // // Combine all data for parallel analysis
    // all_data = gp_with_rand.mix(trials_with_rand)
    
    // // Run all analyses in parallel
    // linear_results = linear_analysis(all_data)
    // trees_results = trees_analysis(all_data)
    // mlp_results = mlp_analysis(all_data)
    
    // // == COMPARISON STAGE ==
    
    // // Prepare channels for joins
    // linear_keyed = linear_results.map { file ->
    //     def base = file.baseName.replace('-LINEAR', '')
    //     tuple(base, file)
    // }
    
    // trees_keyed = trees_results.map { file ->
    //     def base = file.baseName.replace('-TREES', '')
    //     tuple(base, file)
    // }
    
    // mlp_keyed = mlp_results.map { file ->
    //     def base = file.baseName.replace('-MLP', '')
    //     tuple(base, file)
    // }
    
    // // Join results and prepare for comparison
    // comparison_input = linear_keyed
    //     .join(trees_keyed)
    //     .join(mlp_keyed)
    //     .map { base, linear, trees, mlp ->
    //         def analysis = get_analysis_type(base)
    //         tuple(analysis, linear, trees, mlp)
    //     }
    
    // comparisons(comparison_input)
}
