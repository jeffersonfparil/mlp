use chrono::Utc;
use clap::Parser;
use std::env::current_dir;
use std::error::Error;
use std::fs;

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
use crate::network::Network;
use crate::optimisers::{OptimisationParameters, Optimiser, OptimiserError};
use crate::marginal::Marginals;

#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "Simple multilayer perceptron (MLP) from scratch"
)]
struct Args {
    /// Input file name
    #[arg(short = 'f', long)]
    fname: Option<String>,

    /// Delimiter for the input data file
    #[arg(short = 'd', long, default_value = "\t")]
    delim: String,

    /// Vector of column indexes corresponding to the target values in the input data file
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
    #[arg(long, value_parser, value_delimiter = ',', default_value = "128")]
    n_hidden_nodes: Vec<usize>,

    /// Dropout rates per hidden layer
    #[arg(long, value_parser, value_delimiter = ',', default_value = "0.0")]
    dropout_rates: Vec<f32>,

    /// Activation function (Choose from: "ReLU", "Sigmoid", "HyperbolicTangent") (Note: "LeakyReLU" under construction)
    #[arg(long, default_value = "ReLU")]
    activation: String,

    /// Cost function (Choose: "MSE", "MAE", "HL")
    #[arg(long, default_value = "MSE")]
    cost: String,

    /// Optimiser (Choose: "Adam", "AdamMax", "GradientDescent")
    #[arg(long, default_value = "Adam")]
    optimiser: String,

    /// Maximum number of training epochs
    #[arg(long, default_value_t = 10)]
    n_epochs: usize,

    /// Fraction of the maximum number of epochs to wait before enabling the criteria for early stopping
    #[arg(long, default_value_t = 0.25)]
    f_patient_epochs: f32,

    /// Number of training batches to split the input data into
    #[arg(long, default_value_t = 2)]
    n_batches: usize,

    /// Learning rate (η)
    #[arg(long, default_value_t = 0.001)]
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

    /// Filename of the output model (Default: "output_network-{%Y%m%d%H%M%S}.json")
    #[arg(short = 'o', long)]
    fname_network_output: Option<String>,

    /// Verbose
    #[arg(short = 'v', long, action)]
    verbose: bool,

    ////////////////////////////////////////////////////////////////////////////////
    /// Hyperparameter optimisation
    #[arg(long, action)]
    hyperparameter_optimisation: bool,

    /// Range of number of hidden layers for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "1,2,1")]
    range_hidden_layers: Vec<usize>,

    /// Range of number of nodes per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "100,100,100"
    )]
    range_hidden_layer_nodes: Vec<usize>,

    /// Range of dropout rates per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.01"
    )]
    range_dropout_rates: Vec<f32>,

    /// Range of learning rates for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "1e-5,1e-5,1e-5"
    )]
    range_learning_rates: Vec<f32>,

    /// Range of maximum number of training epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "10,10,10")]
    range_n_epochs: Vec<usize>,

    /// Range of proportions of the maximum training epochs to start considering early stopping for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(
        long,
        value_parser,
        value_delimiter = ',',
        default_value = "0.5,1.0,0.5"
    )]
    range_f_patient_epochs: Vec<f32>,

    /// Range of number of batches to split the dataset for hyperparameter optimisation (elements correspond to minimum, maximum and step size)
    #[arg(long, value_parser, value_delimiter = ',', default_value = "1,2,1")]
    range_n_batches: Vec<usize>,

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
        default_value = "GradientDescent,Adam"
    )]
    selection_optimisers: Vec<String>,

    ////////////////////////////////////////////////////////////////////////////////
    /// Predict using a fitted network (fitted MLP model)
    #[arg(long, action)]
    predict_only: bool,

    /// File name of the MLP model in JSON format
    #[arg(short = 'm', long)]
    model: Option<String>,

    ////////////////////////////////////////////////////////////////////////////////
    /// Marginal effects estimation only
    #[arg(short = 'M', long, action)]
    marginals_only: bool,

    // Skip marginal effects estimation
    #[arg(long, action)]
    skip_marginals: bool,

    
    /// Maximum number of interaction effects level, i.e. order 1 includes only the main effects, order 2 includes the main effects and pairwise interactions, and so on
    #[arg(long, default_value_t = 1)]
    marginals_order: usize,
    
    /// Number of input values across the observed range per feature (or input node) to use in predictions
    /// i.e. number of values for interpolate between minimum and maximum values observed in each feature or input node
    #[arg(long, default_value_t = 10)]
    n_interpolate_min_max: usize,

    /// Use DeepSHAP instead of the perturbation method
    /// Note that the current implementation of DeepSHAP generates only main effects and no interaction effects.
    /// Do not enable this flag to use the default perturbation method if you require marginal interaction effects.
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

    /// Two-parameter distribution from which the simulated weights will be sample from
    /// Select from: "normal","lognormal","cauchy","weibull","gamma","beta"
    #[arg(long, default_value = "normal")]
    simulation_weights_distribution: String,

    /// First parameter of the distribution from which the weights will be sampled from
    #[arg(long, default_value_t = 0.0)]
    simulation_weights_distribution_param_1: f64,

    /// First parameter of the distribution from which the weights will be sampled from
    #[arg(long, default_value_t = 1.0)]
    simulation_weights_distribution_param_2: f64,
}

fn read_data(args: &Args) -> Result<Data, Box<dyn Error>> {
    let fname = match &args.fname {
        Some(x) => x.to_owned(),
        None => {
            println!("No input file provided. Simulating data...");
            let data_simulated = Data::simulate(
                args.simulation_n_observations,
                args.simulation_n_features_continuous,
                args.simulation_n_features_categorical.clone(),
                args.simulation_n_output_columns,
                args.simulation_n_hidden_layers,
                &args.simulation_weights_distribution,
                args.simulation_weights_distribution_param_1,
                args.simulation_weights_distribution_param_2,
                args.seed,
            )?;
            let fname_simulated =
                format!("input_simulated-{}.tsv", Utc::now().format("%Y%m%d%H%M%S"));
            data_simulated.write_delimited(&fname_simulated, "\t")?;
            fname_simulated
        }
    };
    Data::read_delimited(&fname, &args.delim, &args.column_indices_of_targets)
}

fn prepare_network(args: &Args, data: &Data) -> Result<Network, Box<dyn Error>> {
    // Simplifying the number of nodes and dropout rates is a single value was entered or left at default
    let n_hidden_layers: usize = args.n_hidden_layers;
    let n_hidden_nodes: Vec<usize> = if (n_hidden_layers > 1) & (args.n_hidden_nodes.len() == 1) {
        vec![args.n_hidden_nodes[0]; n_hidden_layers]
    } else {
        args.n_hidden_nodes.clone()
    };
    let dropout_rates: Vec<f32> = if (n_hidden_layers > 1) & (args.dropout_rates.len() == 1) {
        vec![args.dropout_rates[0]; n_hidden_layers]
    } else {
        args.dropout_rates.clone()
    };
    // Return the network with the input data
    data.init_network(
        n_hidden_layers,
        n_hidden_nodes,
        dropout_rates,
        args.seed,
    )
}

fn simulate_only(args: &Args) -> Result<(), Box<dyn Error>> {
    let data_simulated = Data::simulate(
        args.simulation_n_observations,
        args.simulation_n_features_continuous,
        args.simulation_n_features_categorical.clone(),
        args.simulation_n_output_columns,
        args.simulation_n_hidden_layers,
        &args.simulation_weights_distribution,
        args.simulation_weights_distribution_param_1,
        args.simulation_weights_distribution_param_2,
        args.seed,
    )?;
    let fname_simulated = format!("input_simulated-{}.tsv", Utc::now().format("%Y%m%d%H%M%S"));
    data_simulated.write_delimited(&fname_simulated, "\t")?;
    println!(
        "Please find simulated data: `{}/{}`",
        current_dir()?.display(),
        fname_simulated
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
    let data = read_data(&args)?;
    let network_fitted = Network::read_network(&model)?;
    // Initialise the network containing the input data and fitted model
    let mut network = data.init_network(
        network_fitted.n_hidden_layers,
        network_fitted.n_hidden_nodes.clone(),
        network_fitted.dropout_rates.clone(),
        network_fitted.seed,
    )?;
    network.replace_model(&network_fitted)?;
    // Predict
    network.predict()?;
    // Define the output Data struct containing the prediction
    let n = data.features.n_cols;
    let p = data.features.n_rows;
    let k = data.targets.n_rows + network.predictions.n_rows;
    // println!("n={}; p={}; k={}; data.targets.n_rows={}; network.predictions.n_rows={}", n, p, k, data.targets.n_rows, network.predictions.n_rows);
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
    predictions.targets.data = {
        let y_pred = network.predictions.to_host()?;
        let y_true = data.targets.to_host()?;
        let mut source = vec![f32::NAN; k*n];
        for i in 0..y_pred.len() {
            source[i] = y_pred[i];
        }
        for i in 0..y_true.len() {
            source[y_pred.len() + i] = y_true[i];
        }
        let stream = data.targets.data.context().default_stream();
        stream.clone_htod(&source)?
    };    
    predictions.write_delimited(&fname_predictions, "\t")?;
    println!(
        "Please find the predictions in tab-delimited format: {}/{}",
        current_dir()?.display(),
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
        "Please find the estimated marginal effects in tab-delimited format: {}/{}",
        current_dir()?.display(),
        fname_marginals
    );
    Ok(())
}

fn train_with_hyperparameter_optimisation(
    args: &Args,
    network: &mut Network,
) -> Result<String, Box<dyn Error>> {
    let range_hidden_layers = match args.range_hidden_layers.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of number of hidden layers for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_hidden_layers={:?})",
                    args.range_hidden_layers
                ),
            )));
        }
        false => Some((
            args.range_hidden_layers[0],
            args.range_hidden_layers[1],
            args.range_hidden_layers[2],
        )),
    };
    let range_hidden_layer_nodes = match args.range_hidden_layer_nodes.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of number of nodes per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_hidden_layer_nodes={:?})",
                    args.range_hidden_layer_nodes
                ),
            )));
        }
        false => Some((
            args.range_hidden_layer_nodes[0],
            args.range_hidden_layer_nodes[1],
            args.range_hidden_layer_nodes[2],
        )),
    };
    let range_dropout_rates = match args.range_dropout_rates.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of dropout rates per hidden layer for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_dropout_rates={:?})",
                    args.range_dropout_rates
                ),
            )));
        }
        false => Some((
            args.range_dropout_rates[0],
            args.range_dropout_rates[1],
            args.range_dropout_rates[2],
        )),
    };
    let range_learning_rates = match args.range_learning_rates.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of learning rates for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_learning_rates={:?})",
                    args.range_learning_rates
                ),
            )));
        }
        false => Some((
            args.range_learning_rates[0],
            args.range_learning_rates[1],
            args.range_learning_rates[2],
        )),
    };
    let range_n_epochs = match args.range_n_epochs.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of maximum number of training epochs for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_n_epochs={:?})",
                    args.range_n_epochs
                ),
            )));
        }
        false => Some((
            args.range_n_epochs[0],
            args.range_n_epochs[1],
            args.range_n_epochs[2],
        )),
    };
    let range_f_patient_epochs = match args.range_f_patient_epochs.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of proportions of the maximum training epochs to start considering early stopping for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_f_patient_epochs={:?})",
                    args.range_f_patient_epochs
                ),
            )));
        }
        false => Some((
            args.range_f_patient_epochs[0],
            args.range_f_patient_epochs[1],
            args.range_f_patient_epochs[2],
        )),
    };
    let range_n_batches = match args.range_n_batches.len() != 3 {
        true => {
            return Err(Box::new(OptimiserError::OptimisationParameterError(
                format!(
                    "Range of number of batches to split the dataset for hyperparameter optimisation (elements correspond to minimum, maximum and step size; range_n_batches={:?})",
                    args.range_n_batches
                ),
            )));
        }
        false => Some((
            args.range_n_batches[0],
            args.range_n_batches[1],
            args.range_n_batches[2],
        )),
    };
    let selection_activations: Option<Vec<Activation>> = {
        let mut v: Vec<Activation> = Vec::new();
        for x in &args.selection_activations {
            v.push(match x.as_ref() {
                "ReLU" => Activation::ReLU,
                "Sigmoid" => Activation::Sigmoid,
                "HyperbolicTangent" => Activation::HyperbolicTangent,
                _ => return Err(Box::new(ActivationError::UnimplementedActivation)),
            });
        }
        Some(v)
    };
    let selection_costs: Option<Vec<Cost>> = {
        let mut v: Vec<Cost> = Vec::new();
        for x in &args.selection_costs {
            v.push(match x.as_ref() {
                "MSE" => Cost::MSE,
                "MAE" => Cost::MAE,
                "HL" => Cost::HL,
                _ => return Err(Box::new(CostError::UnimplementedCost)),
            });
        }
        Some(v)
    };
    let selection_optimisers: Option<Vec<Optimiser>> = {
        let mut v: Vec<Optimiser> = Vec::new();
        for x in &args.selection_optimisers {
            v.push(match x.as_ref() {
                "Adam" => Optimiser::Adam,
                "AdamMax" => Optimiser::AdamMax,
                "GradientDescent" => Optimiser::GradientDescent,
                _ => return Err(Box::new(OptimiserError::UnimplementedOptimiser)),
            });
        }
        Some(v)
    };
    let network_hyper_optimised = network.hyperoptimise(
        range_hidden_layers,
        range_hidden_layer_nodes,
        range_dropout_rates,
        range_learning_rates,
        range_n_epochs,
        range_f_patient_epochs,
        range_n_batches,
        selection_activations,
        selection_costs,
        selection_optimisers,
        args.verbose,
    )?;
    // Save the hyperparameter-optimised-trained network
    let fname_network_output = match &args.fname_network_output {
        Some(x) => x.to_owned(),
        None => format!("output_network-{}.json", Utc::now().format("%Y%m%d%H%M%S")),
    };
    network_hyper_optimised.save_network(&fname_network_output)?;
    println!(
        "Please find the output model (network) in json format: {}/{}",
        current_dir()?.display(),
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
    optimisation_parameters.f_patient_epochs = args.f_patient_epochs;
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
        None => format!("output_network-{}.json", Utc::now().format("%Y%m%d%H%M%S")),
    };
    network.save_network(&fname_network_output)?;
    println!(
        "Please find the output model (network) in json format: {}/{}",
        current_dir()?.display(),
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
        "Please find the estimated marginal effects in tab-delimited format: {}/{}",
        current_dir()?.display(),
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
    let data = read_data(&args)?;
    // Prepare the network
    let mut network = prepare_network(&args, &data)?; 
    // Network training and save
    let fname_network_output: String = if args.hyperparameter_optimisation {
        // Perform hyperparameter optimisation then use the best hyperparameters to train the network
        train_with_hyperparameter_optimisation(&args, &mut network)?
    } else {
        // Train the network using the supplied and/or default hyperparameters
        train_with_fixed_hyperparameters(&args, &mut network)?
    };
    // println!("network before saving and reloading: {}", network);
    // Estimate marginal effects after training
    if !args.skip_marginals {
        marginals_after_training(&args, &data, &mut network, fname_network_output)?;
    }
    Ok(())
}
