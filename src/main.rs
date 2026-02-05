use chrono::Utc;
use clap::Parser;
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
    #[arg(short, long)]
    fname: Option<String>,

    /// Delimiter for the input data file
    #[arg(short, long, default_value = "\t")]
    delim: String,

    /// Vector of column indexes corresponding to the target values in the input data file
    #[clap(short, long, value_parser, value_delimiter = ',', default_value = "0")]
    column_indices_of_targets: Vec<usize>,

    /// Number of hidden layers
    #[arg(short, long, default_value_t = 1)]
    n_hidden_layers: usize,

    /// Number of nodes per hidden layer
    #[clap(long, value_parser, value_delimiter = ',', default_value = "128")]
    n_hidden_nodes: Vec<usize>,

    /// Dropout rates per hidden layer
    #[clap(long, value_parser, value_delimiter = ',', default_value = "0.0")]
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
    #[arg(long)]
    fname_network_output: Option<String>,

    /// Verbose
    #[arg(long, action)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!("args.fname: {:?}", args.fname);
    println!("args.delim: {:?}", args.delim);
    println!(
        "args.column_indices_of_targets: {:?}",
        args.column_indices_of_targets
    );

    let fname_network_output = match args.fname_network_output {
        Some(x) => x,
        None => format!("output_network-{}.json", Utc::now().format("%Y%m%d%H%M%S")),
    };

    let fname = match args.fname {
        Some(x) => x,
        None => {
            println!("No input file provided. Simulating data...");
            let data_simulated = Data::simulate(100, 10, 1, 2, 42)?;
            let fname_simulated =
                format!("input_simulated-{}.tsv", Utc::now().format("%Y%m%d%H%M%S"));
            data_simulated.write_delimited(&fname_simulated, "\t")?;
            fname_simulated
        }
    };
    let data = Data::read_delimited(&fname, &args.delim, args.column_indices_of_targets)?;
    println!("data = {}", data);

    let mut network = data.init_network(
        args.n_hidden_layers,
        args.n_hidden_nodes,
        args.dropout_rates,
        args.seed,
    )?;
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
    println!("network = {}", network);

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

    // optimise
    network.train(&optimisation_parameters, args.verbose)?;
    network.save_network(&fname_network_output)?;

    // hyperparameter optim

    Ok(())
}
