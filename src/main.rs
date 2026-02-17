use chrono::Utc;
use clap::Parser;
use std::env::current_dir;
use std::error::Error;

mod activations;
mod backward;
mod costs;
mod forward;
mod io;
mod linalg;
mod network;
mod optimisers;
mod train;

use crate::activations::{Activation, ActivationError};
use crate::costs::{Cost, CostError};
use crate::io::Data;
use crate::network::Network;
use crate::optimisers::{OptimisationParameters, Optimiser, OptimiserError};

/// Simple program to greet a person
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
    predict: bool,

    /// File name of the MLP model in JSON format
    #[arg(short = 'm', long, default_value = "missing-model.json")]
    model: String,

    ////////////////////////////////////////////////////////////////////////////////
    /// Simulate data only
    #[arg(short = 's', long, action)]
    simulate_data_only: bool,

    /// Number of observations to simulate
    #[arg(short = 'n', long, default_value_t = 100)]
    simulation_n_observations: usize,

    /// Number of features to simulate
    #[arg(short = 'p', long, default_value_t = 10)]
    simulation_n_features: usize,

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

fn simulate_only(args: &Args) -> Result<(), Box<dyn Error>> {
    let data_simulated = Data::simulate(
        args.simulation_n_observations,
        args.simulation_n_features,
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
    return Ok(());
}

fn predict_only(args: &Args) -> Result<(), Box<dyn Error>> {
    let fname = match &args.fname {
        Some(x) => x.to_owned(),
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the input data for prediction.",
            )));
        }
    };
    let model = match args.model.as_ref() {
        "missing-model.json" => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Please provide the trained model for prediction (Note that the filename should never be `missing-model.json`).",
            )));
        }
        x => x,
    };
    let mut network = Network::read_input_and_model(
        &fname,
        &args.delim,
        &args.column_indices_of_targets,
        &model,
    )?;
    // Predict
    network.predict()?;
    // Save the updated network containing the fitted weights and biases, targets from the input data, and predictions using the fitted weights and biases
    let fname_network_output = match &args.fname_network_output {
        Some(x) => x.to_owned(),
        None => format!("output_network-{}.json", Utc::now().format("%Y%m%d%H%M%S")),
    };
    network.save_network(&fname_network_output)?;
    // File name of the updated network containing the fitted weights and biases, targets from the input data, and predictions using the fitted weights and biases
    println!("Please find the output model (network)");
    println!("\tcontaining the fitted weights and biases, targets from the input data, and");
    println!(
        "\tpredictions using the fitted weights and biases in json format:\n\t ==> {}/{}",
        current_dir()?.display(),
        fname_network_output
    );
    return Ok(());
}

fn prepare_network_for_training(args: &Args) -> Result<Network, Box<dyn Error>> {
    let fname = match &args.fname {
        Some(x) => x.to_owned(),
        None => {
            println!("No input file provided. Simulating data...");
            let data_simulated = Data::simulate(
                args.simulation_n_observations,
                args.simulation_n_features,
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
    let data = Data::read_delimited(&fname, &args.delim, &args.column_indices_of_targets)?;
    data.init_network(
        args.n_hidden_layers,
        args.n_hidden_nodes.clone(),
        args.dropout_rates.clone(),
        args.seed,
    )
}

fn train_with_hyperparameter_optimisation(
    args: &Args,
    network: &mut Network,
) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

fn train_with_fixed_hyperparameters(
    args: &Args,
    network: &mut Network,
) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse arguments
    let args = Args::parse();
    // Simulate data only
    if args.simulate_data_only {
        return simulate_only(&args);
    }
    // Predict only (using pre-trained model, i.e. in json format)
    if args.predict {
        return predict_only(&args);
    }
    // Load the data including targets and features and output the network for training
    let mut network = prepare_network_for_training(&args)?;
    // Network training
    if args.hyperparameter_optimisation {
        // Perform hyperparameter optimisation then use the best hyperparameters to train the network
        return train_with_hyperparameter_optimisation(&args, &mut network);
    } else {
        // Train the network using the supplied and/or default hyperparameters
        return train_with_fixed_hyperparameters(&args, &mut network);
    }
}
