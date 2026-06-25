use chrono::Utc;
use clap::Parser;
use std::error::Error;
use std::fs;
use std::time::Instant;
use rand::prelude::*;

mod activations;
mod backward;
mod costs;
mod forward;
mod io;
mod linalg;
mod network;
mod optimisers;
mod train;
mod marginal;
mod progress_bar;
mod plot;

use crate::activations::{Activation, ActivationError};
use crate::costs::{Cost, CostError};
use crate::io::Data;
use crate::network::{Network, WeightsInitialisation, NetworkError};
use crate::optimisers::{OptimisationParameters, Optimiser, OptimiserError};
use crate::marginal::Marginals;

fn parse_bound_f32(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("invalid float: {e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("value {v} is out of range [0, 1]"))
    }
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Multi-Layer Perceptron",
    long_about = "A dependency-free Multi-Layer Perceptron (MLP) built from scratch in Rust.\n\n\
    Designed specifically as an alternative to mixed linear models for analysing crop field trial data.\
    It estimates explainable marginal effects, built fundamentally without external ML/DL black-box libraries or genomic data dependencies."
)]
struct Args {
    /// Path to the input dataset
    ///
    /// Delimited file containing the target values (e.g., phenotype data), and predictors 
    /// which can be numeric (binary/continuous) and factor levels. If using PLINK format, 
    /// provide the base name without extensions.
    #[arg(short = 'f', long)]
    fname: Option<String>,

    /// Parse input as a PLINK binary dataset (.bed, .bim, .fam)
    ///
    /// Bypasses standard delimited text parsing to read raw PLINK genetic/phenotypic formats.
    #[arg(long, action)]
    plink: bool,

    /// Delimiter for the input data file
    #[arg(short = 'd', long, default_value = "\t")]
    delim: String,

    /// Zero-based column indices for the target values
    ///
    /// Pass a comma-separated list to specify multiple target phenotypes for multi-task regression.
    #[arg(
        short = 't',
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "0"
    )]
    column_indices_of_targets: Vec<usize>,

    /// Number of hidden layers
    #[arg(long, default_value_t = 1)]
    n_hidden_layers: usize,

    /// Number of nodes per hidden layer
    ///
    /// Comma-separated list matching the length of `n_hidden_layers`.
    #[arg(long, value_parser, value_delimiter = ',', default_value = "64")]
    n_hidden_nodes: Vec<usize>,

    /// Dropout rates per hidden layer
    ///
    /// Comma-separated list of probabilities (0.0 to 1.0) to randomly zero-out nodes during training.
    #[arg(long, value_parser=parse_bound_f32, value_delimiter = ',', default_value = "0.0")]
    dropout_rates: Vec<f32>,

    /// Activation function for hidden nodes
    ///
    /// Options: "ReLU", "Sigmoid", "HyperbolicTangent", "Linear".
    #[arg(long, default_value = "ReLU")]
    activation: String,

    /// Cost/Loss function used to evaluate network error
    ///
    /// Options: "MSE", "MAE", "HL".
    #[arg(long, default_value = "MSE")]
    cost: String,

    /// Optimization algorithm for weight updates
    ///
    /// Options: "Adam", "AdamMax", "GradientDescent".
    #[arg(long, default_value = "Adam")]
    optimiser: String,

    /// Initialisation strategy for layer weights
    ///
    /// Options: "He", "Cauchy", "Uniform", "StandardNormal".
    #[arg(long, default_value = "He")]
    weights_initialisation: String,

    /// Maximum number of training epochs
    #[arg(long, default_value_t = 500)]
    n_epochs: usize,

    /// Number of burnin epochs (initial training epochs to discard)
    #[arg(long, default_value_t = 0)]
    n_burnin_epochs: usize,

    /// Fraction of the maximum number of epochs to wait before enabling the criteria for early stopping
    #[arg(long, value_parser=parse_bound_f32, default_value_t = 0.01)]
    f_patient_epochs: f32,

    /// Fraction of the observations to be used in the estimation of cost at every epoch (using a fixed set of randomly chosen [seeded] observations across all epochs)
    #[arg(long, value_parser=parse_bound_f32, default_value_t = 0.00)]
    f_validation: f32,

    /// Number of training batches to split the input data into
    #[arg(long, default_value_t = 1)]
    n_batches: usize,

    /// Learning rate (η)
    #[arg(long, value_parser=parse_bound_f32, default_value_t = 0.001)]
    learning_rate: f32,

    /// First moment decay (β₁)
    #[arg(long, default_value_t = 0.001)]
    first_moment_decay: f32,

    /// Second moment decay (β₁)
    #[arg(long, default_value_t = 0.999)]
    second_moment_decay: f32,

    /// Small value used for numerical stability (ϵ; usually to avoid dividing by zero)
    #[arg(long, default_value_t = 1e-8)]
    epsilon: f32,

    /// Randomisation seed
    #[arg(long, default_value_t = 123)]
    seed: usize,

    /// Filename of the output network model 
    ///
    /// Saves the trained MLP architecture and weights in JSON format.
    /// Default: "output_network-{%Y%m%d%H%M%S}.json"
    #[arg(short = 'o', long)]
    fname_network_output: Option<String>,

    /// Enable detailed progress logging
    #[arg(short = 'v', long, action)]
    verbose: bool,

    ////////////////////////////////////////////////////////////////////////////////
    /// Hyperparameter optimisation
    #[arg(long, action)]
    hyperparameter_optimisation: bool,

    /// Vector of number of hidden layers for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "1")]
    selection_hidden_layers: Vec<usize>,

    /// Vector of number of nodes per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "1024"
    )]
    selection_hidden_layer_nodes: Vec<usize>,

    /// Vector of dropout rates per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser=parse_bound_f32,
        value_delimiter = ',',
        default_value = "0.0"
    )]
    selection_dropout_rates: Vec<f32>,

    /// Vector of learning rates for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser=parse_bound_f32,
        value_delimiter = ',',
        default_value = "1e-4"
    )]
    selection_learning_rates: Vec<f32>,

    /// Vector of maximum number of training epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "1000,10000")]
    selection_n_epochs: Vec<usize>,

    /// Vector of burnin epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "0")]
    selection_n_burnin_epochs: Vec<usize>,

    /// Vector of proportions of the maximum training epochs to start considering early stopping for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser=parse_bound_f32,
        value_delimiter = ',',
        default_value = "0.1"
    )]
    selection_f_patient_epochs: Vec<f32>,

    /// Vector of proportions of the observations to be used in within training validation set
    #[arg(
        long,
        value_parser=parse_bound_f32,
        value_delimiter = ',',
        default_value = "0.0"
    )]
    selection_f_validation: Vec<f32>,

    /// Vector of number of batches to split the dataset for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "1")]
    selection_n_batches: Vec<usize>,

    /// Activation functions to test
    #[arg(long, value_parser, value_delimiter = ',', default_value = "ReLU")]
    selection_activations: Vec<String>,

    /// Cost functions to test
    #[arg(long, value_parser, value_delimiter = ',', default_value = "MSE")]
    selection_costs: Vec<String>,

    /// Optimisers to test
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "Adam"
    )]
    selection_optimisers: Vec<String>,

    /// Weights initialisations to test
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "He"
    )]
    selection_weights_initialisations: Vec<String>,

    ////////////////////////////////////////////////////////////////////////////////
    /// Execute prediction phase only
    ///
    /// Requires a pre-trained network JSON file passed via `--model`.
    #[arg(long, action)]
    predict_only: bool,

    /// Path to a pre-trained MLP model in JSON format
    #[arg(short = 'm', long)]
    model: Option<String>,

    ////////////////////////////////////////////////////////////////////////////////
    /// Execute only the explainable marginal effects extraction
    ///
    /// Skips training and requires a pre-trained model (JSON file passed via `--model`) to estimate main/interaction effects.
    #[arg(short = 'M', long, action)]
    marginals_only: bool,

    /// Halt execution after training without calculating marginal effects
    #[arg(long, action)]
    skip_marginals: bool,

    /// Maximum interaction level for effects extraction
    ///
    /// Order 1: Main effects only. Order 2: Main + Pairwise interactions. Order 3: Main + Pairwise + Three-way.
    #[arg(long, default_value_t = 1)]
    marginals_order: usize,
    
    /// Number of points to interpolate between observed min/max for each feature
    #[arg(long, default_value_t = 10)]
    n_interpolate_min_max: usize,

    /// Use DeepSHAP for effect estimation instead of the default perturbation method
    ///
    /// Note: DeepSHAP currently only yields main effects. Do not use if interaction effects are required.
    #[arg(short = 'D', long, action)]
    deep_shap: bool,

    /// Number of replications for DeepSHAP main effects estimation
    /// Each replication samples feature values from their normally distributed values
    #[arg(long, default_value_t = 10)]
    deep_shap_reps: usize,
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Simulate data only
    #[arg(short = 's', long, action)]
    simulate_data_only: bool,

    /// Simulated data output filename
    #[arg(long)]
    simulation_fname_output: Option<String>,

    /// Number of observations to simulate
    #[arg(short = 'n', long, default_value_t = 100)]
    simulation_n_observations: usize,

    /// Number of continuous features to simulate
    #[arg(short = 'p', long, default_value_t = 10)]
    simulation_n_features_continuous: usize,
    
    /// Number of continuous features to simulate
    #[arg(
        short = 'q',
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "2,3,5"
    )]
    simulation_n_features_categorical: Vec<usize>,

    /// Number of simulated output column
    #[arg(short = 'k', long, default_value_t = 1)]
    simulation_n_output_columns: usize,

    /// Number of hidden layers to use to simulate the output data
    #[arg(short = 'l', long, default_value_t = 2)]
    simulation_n_hidden_layers: usize,

    /// Distribution from which synthetic network weights are sampled
    ///
    /// Options: "normal", "lognormal", "cauchy", "weibull", "gamma", "beta".
    #[arg(long, default_value = "normal")]
    simulation_weights_distribution: String,

    /// Parameter 1 (e.g., mean or location or shape) for the weight distribution
    #[arg(long, default_value_t = 0.0)]
    simulation_weights_distribution_param_1: f64,

    /// Parameter 2 (e.g., variance or scale) for the weight distribution
    #[arg(long, default_value_t = 1.0)]
    simulation_weights_distribution_param_2: f64,

    ////////////////////////////////////////////////////////////////////////////////
    // Miscellaneous flags and arguments

    /// Do not save the network (for benchmarking purposes to save on time and resources by not writing the model as JSON into disk)
    #[arg(long, action)]
    do_not_save_network: bool,
}

fn read_data(args: &Args) -> Result<Data, Box<dyn Error>> {
    let fname = match &args.fname {
        Some(x) => x.to_owned(),
        None => {
            println!("No input file provided. Simulating data...");
            let (data_simulated, network_simulated) = Data::simulate(
                args.simulation_n_observations,
                args.simulation_n_features_continuous,
                args.simulation_n_features_categorical.clone(),
                args.simulation_n_output_columns,
                args.simulation_n_hidden_layers,
                &args.simulation_weights_distribution,
                args.simulation_weights_distribution_param_1,
                args.simulation_weights_distribution_param_2,
                args.seed,
                args.verbose,
            )?;
            let mut rng = rand::rng();
            let fname_simulated = format!(
                    "input_simulated-T{}-R{}.tsv", 
                    Utc::now().format("%Y%m%d%H%M%S"),
                    rng.random::<u32>(),
            );
            data_simulated.write_delimited(&fname_simulated, "\t")?;
            fname_simulated
        }
    };
    if args.plink {
        return Data::from_plink(&fname)
    } else {
        return Data::read_delimited(&fname, &args.delim, &args.column_indices_of_targets)
    }
}

fn prepare_network(args: &Args, data: &Data) -> Result<Network, Box<dyn Error>> {
    // Simplifying the number of nodes and dropout rates is a single value was entered or left at default
    let n_hidden_layers: usize = args.n_hidden_layers;
    let n_hidden_nodes: Vec<usize> = if (n_hidden_layers > 1) & (args.n_hidden_nodes.len() == 1) {
        vec![args.n_hidden_nodes[0]; n_hidden_layers]
    } else {
        args.n_hidden_nodes.clone()
    };
    if n_hidden_nodes.len() != n_hidden_layers {
        return Err(Box::new(NetworkError::OtherError(format!("The number of supplied values of hidden nodes ({:?}) is not equal to the number of hidden layers ({})", n_hidden_nodes, n_hidden_layers))))
    }
    let dropout_rates: Vec<f32> = if (n_hidden_layers > 1) & (args.dropout_rates.len() == 1) {
        vec![args.dropout_rates[0]; n_hidden_layers]
    } else {
        args.dropout_rates.clone()
    };
    if dropout_rates.len() != n_hidden_layers {
        return Err(Box::new(NetworkError::OtherError(format!("The number of supplied values of dropout rates ({:?}) is not equal to the number of hidden layers ({})", dropout_rates, n_hidden_layers))))
    }
    let weights_initialisation = match args.weights_initialisation.as_ref() {
        "He" => WeightsInitialisation::He,
        "Cauchy" => WeightsInitialisation::Cauchy,
        "Uniform" => WeightsInitialisation::Uniform,
        "StandardNormal" => WeightsInitialisation::StandardNormal,
        e => return Err(Box::new(NetworkError::OtherError(format!("Unrecognised weights initialisation: {}", e)))),
    };
    // Return the initialised network
    data.init_network(
        n_hidden_layers,
        n_hidden_nodes,
        dropout_rates,
        weights_initialisation,
        args.seed,
    )
    // Re-initialise the weights
    
}

fn simulate_only(args: &Args) -> Result<(), Box<dyn Error>> {
    let (data_simulated, network_simulated) = Data::simulate(
        args.simulation_n_observations,
        args.simulation_n_features_continuous,
        args.simulation_n_features_categorical.clone(),
        args.simulation_n_output_columns,
        args.simulation_n_hidden_layers,
        &args.simulation_weights_distribution,
        args.simulation_weights_distribution_param_1,
        args.simulation_weights_distribution_param_2,
        args.seed,
        args.verbose,
    )?;
    let fname_simulated = match &args.simulation_fname_output {
        Some(x) => x.to_owned(),
        None => {
            let mut rng = rand::rng();
            format!("input_simulated-n_{}-p_{}-q_{}-k_{}-d_{}-D{:?}-Dp1_{}-Dp1_{}-s_{}-t_{}-r_{}.tsv", 
                args.simulation_n_observations,
                args.simulation_n_features_continuous,
                args.simulation_n_features_categorical.iter().fold(0, |sum, x| sum + x),
                args.simulation_n_output_columns,
                args.simulation_n_hidden_layers,
                &args.simulation_weights_distribution,
                args.simulation_weights_distribution_param_1,
                args.simulation_weights_distribution_param_2,
                args.seed,
                Utc::now().format("%Y%m%d%H%M%S"),
                rng.random::<u32>(),
            )
        }
    };
    let fname_simulated_network = fname_simulated.replace(".tsv", ".json");
    data_simulated.write_delimited(&fname_simulated, "\t")?;
    network_simulated.save_network(&fname_simulated_network)?;
    println!(
        "Please find simulated data: `{}` and simulated network: `{}`",
        fname_simulated, fname_simulated_network,
    );
    Ok(())
}

fn predict_only(args: &Args) -> Result<(), Box<dyn Error>> {
    match &args.fname {
        Some(_) => (),
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the input data for prediction.",
            )));
        }
    };
    let model = match &args.model {
        Some(x) => x.to_owned(),
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the trained model for prediction.",
            )));
        }
    };
    let fname_predictions = model.replace(".json", "-predictions.tsv");
    match fs::File::create_new(&fname_predictions) {
        Ok(_) => {std::fs::remove_file(&fname_predictions)?},
        Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Predictions file '{}' exists!", fname_predictions)))),
    }
    // Prepare the network
    // Load input data
    if args.verbose {println!("(1/5) Loading input data...")}; let time = Instant::now();
    let data = read_data(&args)?;
    if args.verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
    if args.verbose {println!("(2/5) Loading model...")}; let time = Instant::now();
    let network_fitted = Network::read_network(&model)?;
    if args.verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
    // Initialise the network containing the input data and fitted model
    if args.verbose {println!("(3/5) Preparing the network...")}; let time = Instant::now();
    let mut network = data.init_network(
        network_fitted.n_hidden_layers,
        network_fitted.n_hidden_nodes.clone(),
        network_fitted.dropout_rates.clone(),
        network_fitted.weights_initialisation.clone(),
        network_fitted.seed,
    )?;
    network.replace_model(&network_fitted)?;
    if args.verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
    // Predict
    if args.verbose {println!("(4/5) Predicting...")}; let time = Instant::now();
    network.predict()?;
    if args.verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
    // Define the output Data struct containing the prediction
    if args.verbose {println!("(5/5) Saving the predictions...")}; let time = Instant::now();
    let n = data.features.n_cols;
    let p = data.features.n_rows;
    let k = data.targets.n_rows + network.predictions.n_rows;
    // println!("n={}; p={}; k={}; data.targets.n_rows={}; network.predictions.n_rows={}", n, p, k, data.targets.n_rows, network.predictions.n_rows);
    // First column/s is/are the predictions and the rest are the observed/expected values
    let mut predictions = Data::new(n, p, k)?;
    predictions.feature_names = data.feature_names.clone();
    predictions.features = data.features.clone();
    predictions.target_names = {
        let mut target_names: Vec<String> = vec!["".to_owned(); k];
        for i in 0..data.targets.n_rows {
            target_names[i] = format!("predict-{}", data.target_names[i]);
        }
        for i in 0..data.targets.n_rows {
            target_names[data.targets.n_rows + i] = data.target_names[i].to_owned();
        }
        target_names
    };
    // Use the mean and sd to unstandardise the predictions
    let mu: f32 = network.targets_mean_sd.0;
    let sd: f32 = network.targets_mean_sd.1;
    predictions.targets.data = {
        let y_pred = network.predictions.to_host()?;
        let y_true = data.targets.to_host()?;
        let mut source = vec![f32::NAN; k*n];
        for i in 0..y_pred.len() {
            source[i] = (y_pred[i] * sd) + mu; // unstandardise
        }
        for i in 0..y_true.len() {
            source[y_pred.len() + i] = (y_true[i] * sd) + mu; // unstandardise
        }
        let stream = data.targets.data.context().default_stream();
        stream.clone_htod(&source)?
    };

    predictions.write_delimited(&fname_predictions, "\t")?;
    if args.verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
    println!(
        "Please find the predictions in tab-delimited format: {}",
        fname_predictions
    );
    Ok(())
}

fn marginals_only(args: &Args) -> Result<(), Box<dyn Error>> {
    match &args.fname {
        Some(_) => (),
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the input data used in training the model because it is needed to instantiate the marginal effects ids.",
            )));
        }
    };
    let model = match &args.model {
        Some(x) => x.to_owned(),
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the trained model for marginal effects estimation.",
            )));
        }
    };
    match args.marginals_order < 1 {
        true => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Maximum interaction effects level/order cannot be less than 1.",
            )));
        },
        false => (),
    };
    let fname_marginals = model.replace(".json", "-marginal_effects.tsv");
    match fs::File::create_new(&fname_marginals) {
        Ok(_) => {std::fs::remove_file(&fname_marginals)?},
        Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Marginal effects file '{}' exists!", fname_marginals)))),
    }
    // Load the data including targets and features
    let data = read_data(&args)?;
    match args.marginals_order > data.feature_names.len() {
        true => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Maximum interaction effects level/order greater than the number of features ({}).", data.feature_names.len()),
            )));
        },
        false => (),
    };
    // Prepare the network
    let mut network = Network::read_network(&model)?;
    // println!("network after saving and reloading: {}", network);
    // Extract the marginal effects and save
    // Note that the maximum order of effects is naively set to `network.n_hidden_layers + 1` even though technically all possible feature combinations are possible even at 1 hidden layer
    let marginals_order: usize = if args.marginals_order > data.feature_names.len() {
        data.feature_names.len()
    } else {
        args.marginals_order
    };
    let mut marginals = Marginals::new(data.feature_names.clone(), marginals_order)?;
    if !args.deep_shap {
        // Use perturbation analysis to estimate main and interaction (if requested) effects
        marginals.estimate_perturb(&mut network, args.n_interpolate_min_max, args.verbose)?;
    } else {
        // Use DeepSHAP to estimate main effects only
        marginals.estimate_deepshap(&mut network, args.deep_shap_reps, args.seed, args.verbose)?;
    }
    marginals.write_delimited(&fname_marginals, "\t")?;
    println!(
        "Please find the estimated marginal effects in tab-delimited format: {}",
        fname_marginals
    );
    Ok(())
}

fn train_with_hyperparameter_optimisation(
    args: &Args,
    network: &mut Network,
) -> Result<String, Box<dyn Error>> {
    let selection_activations: Vec<Activation> = {
        let mut v: Vec<Activation> = Vec::new();
        for x in &args.selection_activations {
            v.push(match x.as_ref() {
                "ReLU" => Activation::ReLU,
                "Sigmoid" => Activation::Sigmoid,
                "HyperbolicTangent" => Activation::HyperbolicTangent,
                "Linear" => Activation::Linear,
                _ => return Err(Box::new(ActivationError::UnimplementedActivation)),
            });
        }
        v
    };
    let selection_costs: Vec<Cost> = {
        let mut v: Vec<Cost> = Vec::new();
        for x in &args.selection_costs {
            v.push(match x.as_ref() {
                "MSE" => Cost::MSE,
                "MAE" => Cost::MAE,
                "HL" => Cost::HL,
                _ => return Err(Box::new(CostError::UnimplementedCost)),
            });
        }
        v
    };
    let selection_optimisers: Vec<Optimiser> = {
        let mut v: Vec<Optimiser> = Vec::new();
        for x in &args.selection_optimisers {
            v.push(match x.as_ref() {
                "Adam" => Optimiser::Adam,
                "AdamMax" => Optimiser::AdamMax,
                "GradientDescent" => Optimiser::GradientDescent,
                _ => return Err(Box::new(OptimiserError::UnimplementedOptimiser)),
            });
        }
        v
    };
    let selection_weights_initialisations: Vec<WeightsInitialisation> = {
        let mut v: Vec<WeightsInitialisation> = Vec::new();
        for x in &args.selection_weights_initialisations {
            v.push(match x.as_ref() {
                "He" => WeightsInitialisation::He,
                "Cauchy" => WeightsInitialisation::Cauchy,
                "Uniform" => WeightsInitialisation::Uniform,
                "StandardNormal" => WeightsInitialisation::StandardNormal,
                e => return Err(Box::new(NetworkError::OtherError(format!("Unrecognised weights initialisation: {}", e)))),
            });
        }
        v
    };
    let network_hyper_optimised = network.hyperoptimise(
        &args.selection_hidden_layers,
        &args.selection_hidden_layer_nodes,
        &args.selection_dropout_rates,
        &args.selection_learning_rates,
        &args.selection_n_epochs,
        &args.selection_n_burnin_epochs,
        &args.selection_f_patient_epochs,
        &args.selection_f_validation,
        &args.selection_n_batches,
        &selection_activations,
        &selection_costs,
        &selection_optimisers,
        &selection_weights_initialisations,
        args.verbose,
    )?;
    // Save the hyperparameter-optimised-trained network
    let fname_network_output = match &args.fname_network_output {
        Some(x) => x.to_owned(),
        None => {
            let mut rng = rand::rng();
            format!("output_network-T{}-R{}.json", Utc::now().format("%Y%m%d%H%M%S"), rng.random::<u32>(),)
        },
    };
    network_hyper_optimised.save_network(&fname_network_output)?;
    println!(
        "Please find the output model (network) in json format: {}",
        fname_network_output
    );
    Ok(fname_network_output)
}

fn train_with_fixed_hyperparameters(
    args: &Args,
    network: &mut Network,
) -> Result<String, Box<dyn Error>> {
    network.activation = match args.activation.as_ref() {
        "ReLU" => Activation::ReLU,
        "Sigmoid" => Activation::Sigmoid,
        "HyperbolicTangent" => Activation::HyperbolicTangent,
        "Linear" => Activation::Linear,
        _ => return Err(Box::new(ActivationError::UnimplementedActivation)),
    };
    network.cost = match args.cost.as_ref() {
        "MSE" => Cost::MSE,
        "MAE" => Cost::MAE,
        "HL" => Cost::HL,
        _ => return Err(Box::new(CostError::UnimplementedCost)),
    };
    let mut optimisation_parameters = OptimisationParameters::new(&network)?;
    optimisation_parameters.optimiser = match args.optimiser.as_ref() {
        "Adam" => Optimiser::Adam,
        "AdamMax" => Optimiser::AdamMax,
        "GradientDescent" => Optimiser::GradientDescent,
        _ => return Err(Box::new(OptimiserError::UnimplementedOptimiser)),
    };
    optimisation_parameters.n_epochs = args.n_epochs;
    optimisation_parameters.n_burnin_epochs = args.n_burnin_epochs;
    optimisation_parameters.f_patient_epochs = args.f_patient_epochs;
    optimisation_parameters.f_validation = args.f_validation;
    optimisation_parameters.n_batches = args.n_batches;
    optimisation_parameters.learning_rate = args.learning_rate;
    optimisation_parameters.first_moment_decay = args.first_moment_decay;
    optimisation_parameters.second_moment_decay = args.second_moment_decay;
    optimisation_parameters.epsilon = args.epsilon;
    // Train
    network.train(&optimisation_parameters, args.verbose)?;
    // Save the trained network
    let fname_network_output = match &args.fname_network_output {
        Some(x) => x.to_owned(),
        None => {
            let mut rng = rand::rng();
            format!("output_network-T{}-R{}.json", Utc::now().format("%Y%m%d%H%M%S"), rng.random::<u32>(),)
        },
    };
    network.save_network(&fname_network_output)?;
    println!(
        "Please find the output model (network) in json format: {}",
        fname_network_output
    );
    Ok(fname_network_output)
}

fn marginals_after_training(
    args: &Args,
    data: &Data,
    network: &mut Network,
    fname_network_output: String
) -> Result<(), Box<dyn Error>> {
    let fname_marginals = fname_network_output.replace(".json", "-marginal_effects.tsv");
    match fs::File::create_new(&fname_marginals) {
        Ok(_) => {std::fs::remove_file(&fname_marginals)?},
        Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Marginal effects file '{}' exists!", fname_marginals)))),
    }
    match args.marginals_order < 1 {
        true => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Maximum interaction effects level/order cannot be less than 1.",
            )));
        },
        false => (),
    };
    match args.marginals_order > data.feature_names.len() {
        true => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Maximum interaction effects level/order greater than the number of features ({}).", data.feature_names.len()),
            )));
        },
        false => (),
    };
    // Extract the marginal effects and save
    // Note that the maximum order of effects is naively set to `network.n_hidden_layers + 1` even though technically all possible feature combinations are possible even at 1 hidden layer
    let marginals_order: usize = if args.marginals_order > data.feature_names.len() {
        data.feature_names.len()
    } else {
        args.marginals_order
    };
    let mut marginals = Marginals::new(data.feature_names.clone(), marginals_order)?;
    if !args.deep_shap {
        // Use perturbation analysis to estimate main and interaction (if requested) effects
        marginals.estimate_perturb(network, args.n_interpolate_min_max, args.verbose)?;
    } else {
        // Use DeepSHAP to estimate main effects only
        marginals.estimate_deepshap(network, args.deep_shap_reps, args.seed, args.verbose)?;
    }
    marginals.write_delimited(&fname_marginals, "\t")?;
    println!(
        "Please find the estimated marginal effects in tab-delimited format: {}",
        fname_marginals
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments
    let args = Args::parse();
    // Make sure the output and input models if they are supplied have the extension "json"
    // which is required to make sure the marginal effects filenames are generated correctly
    match &args.fname_network_output {
        Some(x) => {
            let path = std::path::Path::new(x);
            if path.extension().map_or(false, |ext| ext != "json") {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Output model file needs to have '.json' extension: {}", x))))
            }
            if path.exists() {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Output model file already exists: {}", x))))
            }
        },
        None => (),
    };
    match &args.model {
        Some(x) => {
            let path = std::path::Path::new(x);
            if path.extension().map_or(false, |ext| ext != "json") {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Input model file needs to have '.json' extension: {}", x))))
            }
        },
        None => (),
    };
    match &args.simulation_fname_output {
        Some(x) => {
            let path = std::path::Path::new(x);
            if path.exists() {
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Simulated data file already exists: {}", x))))
            }
        },
        None => (),
    };
    // Simulate data only
    if args.simulate_data_only {
        return simulate_only(&args)
    }
    // Predict only (using pre-trained model, i.e. in json format)
    if args.predict_only {
        return predict_only(&args)
    }
    // Marginal effects estimation only (using pre-trained model, i.e. in json format)
    if args.marginals_only {
        return marginals_only(&args)
    }
    // Load the data including targets and features
    let n_steps: usize = if args.skip_marginals {3} else {4};
    if args.verbose {println!("(1/{}) Loading data...", n_steps)};
    let data = read_data(&args)?;
    // Prepare the network
    if args.verbose {println!("(2/{}) Preparing network...", n_steps)};
    let mut network = prepare_network(&args, &data)?; 
    // Network training and save
    let fname_network_output: String = if args.hyperparameter_optimisation {
        // Perform hyperparameter optimisation then use the best hyperparameters to train the network
        if args.verbose {println!("(3/{}) Training with hyperparameter optimisation...", n_steps)};
        train_with_hyperparameter_optimisation(&args, &mut network)?
    } else {
        // Train the network using the supplied and/or default hyperparameters
        if args.verbose {println!("(3/{}) Training with user-supplied/default hyperparameters...", n_steps)};
        train_with_fixed_hyperparameters(&args, &mut network)?
    };
    // println!("network before saving and reloading: {}", network);
    // Estimate marginal effects after training
    if !args.skip_marginals {
        if args.verbose {println!("(4/{}) Extracting marginal effects...", n_steps)};
        marginals_after_training(&args, &data, &mut network, fname_network_output)?;
    }
    Ok(())
}
