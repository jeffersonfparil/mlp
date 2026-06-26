use crate::activations::Activation;
use crate::costs::Cost;
use crate::network::{Network, WeightsInitialisation};
use crate::optimisers::{OptimisationParameters, Optimiser};
use crate::progress_bar::ProgressBar;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
// use rand_distr::weighted::Weight;
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Debug, PartialEq)]
pub enum TrainingError {
    BatchingError(String),
    EpochError(String),
    OtherError(String),
}

/// Implement Error for TrainingError
impl Error for TrainingError {}

/// Implement std::fmt::Display for TrainingError
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingError::BatchingError(msg) => {
                write!(f, "Batching error during training: {}", msg)
            }
            TrainingError::EpochError(msg) => write!(f, "Epoch error during training: {}", msg),
            TrainingError::OtherError(msg) => write!(f, "Other error during training: {}", msg),
        }
    }
}

fn prep_all_hyperparams(
    selection_hidden_layers: &Vec<usize>,
    selection_hidden_layer_nodes: &Vec<usize>,
    selection_dropout_rates: &Vec<f32>,
    selection_learning_rates: &Vec<f32>,
    selection_n_epochs: &Vec<usize>,
    selection_n_burnin_epochs: &Vec<usize>,
    selection_f_patient_epochs: &Vec<f32>,
    selection_f_validation: &Vec<f32>,
    selection_n_batches: &Vec<usize>,
    selection_activations: &Vec<Activation>,
    selection_costs: &Vec<Cost>,
    selection_optimisers: &Vec<Optimiser>,
    selection_weights_initialisations: &Vec<WeightsInitialisation>,
) -> Result<
    Vec<(
        usize,
        usize,
        f32,
        f32,
        usize,
        usize,
        f32,
        f32,
        usize,
        Activation,
        Cost,
        Optimiser,
        WeightsInitialisation,
    )>,
    Box<dyn Error>,
> {
    let selection_hidden_layers: Vec<usize> = selection_hidden_layers.clone().into_iter().collect::<HashSet<_>>().into_iter().collect();
    let selection_hidden_layer_nodes: Vec<usize> = selection_hidden_layer_nodes.clone().into_iter().collect::<HashSet<_>>().into_iter().collect();
    let selection_dropout_rates: Vec<f32> = {
        let mut rates = selection_dropout_rates.clone();
        rates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        rates.dedup_by(|a, b| a == b);
        if rates.iter().any(|&r| !(0.0..=1.0).contains(&r)) {
            return Err(Box::new(TrainingError::OtherError(format!(
                "Drop-out rates out of bounds: {:?}",
                rates
            ))));
        }
        rates
    };
    let selection_learning_rates: Vec<f32> = {
        let mut rates = selection_learning_rates.clone();
        rates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        rates.dedup_by(|a, b| a == b);
        if rates.iter().any(|&r| !(0.0..=1.0).contains(&r)) {
            return Err(Box::new(TrainingError::OtherError(format!(
                "Learning rates out of bounds: {:?}",
                rates
            ))));
        }
        rates
    };
    let selection_n_epochs: Vec<usize> = selection_n_epochs.clone().into_iter().collect::<HashSet<_>>().into_iter().collect();
    let selection_n_burnin_epochs: Vec<usize> = selection_n_burnin_epochs.clone().into_iter().collect::<HashSet<_>>().into_iter().collect();
    let selection_f_patient_epochs: Vec<f32> = {
        let mut fracs = selection_f_patient_epochs.clone();
        fracs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        fracs.dedup_by(|a, b| a == b);
        if fracs.iter().any(|&r| !(0.0..=1.0).contains(&r)) {
            return Err(Box::new(TrainingError::OtherError(format!(
                "Fractions of patient epochs out of bounds: {:?}",
                fracs
            ))));
        }
        fracs
    };
    let selection_f_validation: Vec<f32> = {
        let mut fracs = selection_f_validation.clone();
        fracs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        fracs.dedup_by(|a, b| a == b);
        if fracs.iter().any(|&r| !(0.0..=1.0).contains(&r)) {
            return Err(Box::new(TrainingError::OtherError(format!(
                "Fractions of patient epochs out of bounds: {:?}",
                fracs
            ))));
        }
        fracs
    };
    let selection_n_batches: Vec<usize> = selection_n_batches.clone().into_iter().collect::<HashSet<_>>().into_iter().collect();
    let selection_activations: Vec<Activation> = selection_activations
        .clone()
        .into_iter()
        .fold(Vec::new(), |mut acc, item| {
            if !acc.contains(&item) {
                acc.push(item);
            }
            acc
        });
    let selection_costs: Vec<Cost> = selection_costs
        .clone()
        .into_iter()
        .fold(Vec::new(), |mut acc, item| {
            if !acc.contains(&item) {
                acc.push(item);
            }
            acc
        });
    let selection_optimisers: Vec<Optimiser> = selection_optimisers
        .clone()
        .into_iter()
        .fold(Vec::new(), |mut acc, item| {
            if !acc.contains(&item) {
                acc.push(item);
            }
            acc
        });
    let selection_weights_initialisations: Vec<WeightsInitialisation> = selection_weights_initialisations
        .clone()
        .into_iter()
        .fold(Vec::new(), |mut acc, item| {
            if !acc.contains(&item) {
                acc.push(item);
            }
            acc
        });
    let mut param_combinations: Vec<(
        usize,
        usize,
        f32,
        f32,
        usize,
        usize,
        f32,
        f32,
        usize,
        Activation,
        Cost,
        Optimiser,
        WeightsInitialisation,
    )> = Vec::new();
    for n_hidden_layers in &selection_hidden_layers {
        for n_hidden_nodes in &selection_hidden_layer_nodes {
            for dropout_rate in &selection_dropout_rates {
                for learning_rate in &selection_learning_rates {
                    for n_epochs in &selection_n_epochs {
                        for n_burnin_epochs in &selection_n_burnin_epochs {
                            for f_patient_epochs in &selection_f_patient_epochs {
                                for f_validation in &selection_f_validation {
                                    for n_batches in &selection_n_batches {
                                        for activation in &selection_activations {
                                            for cost in &selection_costs {
                                                for optimiser in &selection_optimisers {
                                                    for weights_initialisation in &selection_weights_initialisations {
                                                        param_combinations.push((
                                                            *n_hidden_layers,
                                                            *n_hidden_nodes,
                                                            *dropout_rate,
                                                            *learning_rate,
                                                            *n_epochs,
                                                            *n_burnin_epochs,
                                                            *f_patient_epochs,
                                                            *f_validation,
                                                            *n_batches,
                                                            activation.clone(),
                                                            cost.clone(),
                                                            optimiser.clone(),
                                                            weights_initialisation.clone(),
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(param_combinations)
}

impl Network {
    pub fn shufflesplit(self: &Self, n_batches: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let n: usize = self.targets.n_cols; // number of observations
        if n_batches == 0 {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        if n_batches > n {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches cannot be greater than number of observations.".to_string(),
            )));
        }
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);
        let batch_size = (n + n_batches - 1) / n_batches;
        let mut col_indexes_per_batch: Vec<Vec<usize>> = Vec::new();
        for i in 0..n_batches {
            let start = i * batch_size;
            let end = match (i + 1) * batch_size {
                x if x > n => n,
                x => x,
            };
            if start >= end {
                break;
            }
            col_indexes_per_batch.push(indexes[start..end].to_vec());
        }
        Ok(col_indexes_per_batch)
    }

    pub fn train_per_batch(
        self: &mut Self,
        optimisation_parameters: &mut OptimisationParameters,
        n_batches: &str,
        verbose: bool,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let mut epochs: Vec<f64> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();
        let n_patient_epochs = if (optimisation_parameters.f_patient_epochs < 0.0) | (optimisation_parameters.f_patient_epochs > 1.0) {
            return Err(Box::new(TrainingError::OtherError(format!("The fraction of patient epochs is out of bounds: {}", optimisation_parameters.f_patient_epochs))))
        } else {
            (optimisation_parameters.f_patient_epochs
            * (optimisation_parameters.n_epochs - 1) as f32)
            .floor() as usize + 1
        };
        // Note that for large networks this can be very VRAM-hungry! TODO: make this more efficient probably for non-vross-validating runs
        // With or without cross-validation
        let n: usize = self.targets.n_cols;
        let n_validation: usize = if (optimisation_parameters.f_validation < 0.0) | (optimisation_parameters.f_validation > 1.0) {
            return Err(Box::new(TrainingError::OtherError(format!("The fraction of observations for validation is out of bounds: {}", optimisation_parameters.f_validation))))
        } else {
            (n as f32 * optimisation_parameters.f_validation).floor() as usize
        };
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let validation_indexes: Vec<usize> = (0..n).choose_multiple(&mut rng, n_validation);
        let training_indexes: Vec<usize> = (0..n)
            .filter(|&x| !validation_indexes.contains(&x))
            .collect();
        let (mut network_validation, mut network_training) = if n_validation > 0 {
            (self.slice(&validation_indexes)?,  self.slice(&training_indexes)?)
        } else {
            (self.slice(&vec![0])?, self.slice(&training_indexes)?)
        };
        // Pre-training burn-in epochs
        let mut pb = ProgressBar::new(optimisation_parameters.n_burnin_epochs, 50, format!("Burn-in {} epochs", optimisation_parameters.n_burnin_epochs));
        for _ in 0..optimisation_parameters.n_burnin_epochs {
            network_training.forwardpass()?;
            network_training.backpropagation()?;
            network_training.optimise(optimisation_parameters)?;
            network_training.predict()?;
            if verbose {
                pb.next();
            }
        }
        if verbose {
            pb.finish();
        }
        // Training after burn-in
        let mut pb = ProgressBar::new(optimisation_parameters.n_epochs, 50, format!("Training {} batches (seed={}, nt={}, nv={})", n_batches, self.seed, n-n_validation, n_validation));
        for epoch in 0..optimisation_parameters.n_epochs {
            network_training.forwardpass()?;
            network_training.backpropagation()?;
            network_training.optimise(optimisation_parameters)?;
            network_training.predict()?;
            epochs.push(epoch as f64);
            // Validate
            if n_validation > 0 {
                network_validation.replace_model(&network_training)?;
                network_validation.predict()?;
                costs.push(network_validation.loss()? as f64);
            } else {
                costs.push(network_training.loss()? as f64);
            }
            // Update the network after training the training network
            if verbose {
                pb.next();
            }
            // Early stopping check, i.e. stop if no improvement in cost after n_patient_epochs
            if (epoch > n_patient_epochs) && (costs[epoch] >= costs[epoch - n_patient_epochs]) {
                // println!("Early stopping at epoch {}", epoch);
                break;
            }
       }
        // Update the network after training the training network
        self.replace_model(&network_training)?;
        if verbose {
            pb.finish();
        }
        self.predict()?;
        self.n_epochs = epochs.len();
        Ok((epochs, costs))
    }

    pub fn train(
        self: &mut Self,
        optimisation_parameters: &OptimisationParameters,
        verbose: bool,
    ) -> Result<f32, Box<dyn Error>> {
        self.check_dimensions()?;
        if optimisation_parameters.n_epochs == 0 {
            return Err(Box::new(TrainingError::EpochError(
                "Number of epochs must be greater than zero.".to_string(),
            )));
        }
        if optimisation_parameters.n_batches == 0 {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        let (epochs, costs): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if optimisation_parameters.n_batches == 1 {
            // Only one batch, train on the whole dataset
            let mut params = optimisation_parameters.clone();
            let (epochs, costs) = self.train_per_batch(&mut params, "1", verbose)?;
            // self.predict()?;
            (vec![epochs], vec![costs])
        } else {
            // Multiple batches, split the dataset then average the parameters after training on each batch
            let col_indexes_per_batch: Vec<Vec<usize>> =
                self.shufflesplit(optimisation_parameters.n_batches)?;
            let mut networks_per_batch: Vec<Network> =
                Vec::with_capacity(optimisation_parameters.n_batches);
            for col_indexes in col_indexes_per_batch {
                // indexes for each batch, i.e. for observations
                let network: Network = self.slice(&col_indexes)?;
                networks_per_batch.push(network);
            }
            // let epochs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
            // let costs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
            let n = networks_per_batch.len();
            let epochs: Mutex<Vec<Vec<f64>>> = Mutex::new(vec![Vec::new(); n]);
            let costs: Mutex<Vec<Vec<f64>>> = Mutex::new(vec![Vec::new(); n]);
            networks_per_batch
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, network)| {
                    if verbose {
                        println!(
                            "Training on batch {} with {} observations.",
                            i, network.targets.n_cols
                        );
                    }
                    let mut params = optimisation_parameters.clone();
                    let result = network.train_per_batch(&mut params, &format!("{}", n), verbose);
                    match result {
                        Ok((epochs_batch, costs_batch)) => {
                            // epochs.lock().unwrap().push(epochs_batch);
                            // costs.lock().unwrap().push(costs_batch);
                            // Lock the mutexes to get access
                            let mut locked_epochs = epochs.lock().unwrap();
                            let mut locked_costs = costs.lock().unwrap();
                            // Replace the ith element safely
                            if let Some(x) = locked_epochs.get_mut(i) {
                                *x = epochs_batch;
                            }
                            if let Some(x) = locked_costs.get_mut(i) {
                                *x = costs_batch;
                            }
                        }
                        Err(e) => {
                            // Skip the batch
                            eprintln!("Error training on batch {}: {}", i+1, e);
                        }
                    }
                }
            );
            // Merge the parameters from each batch network back into the original network via simple averaging with a better method
            self.average_weights_biases(&networks_per_batch)?;
            // Return epochs, costs
            (epochs.into_inner().unwrap(), costs.into_inner().unwrap())
        };
        // Assess cost after training
        let final_cost_value = self.loss()?;
        if verbose {
            let fname_loss_svg = self.plot_loss(epochs, costs, optimisation_parameters)?;
            let fname_scatter_svg = self.plot_true_vs_pred(optimisation_parameters)?;
            println!("===============================================");
            println!("Final cost after training: {}", final_cost_value);
            println!("Find the loss curve saved as: {}", fname_loss_svg);
            println!("Find the observed vs predicted scatterplot saved as: {}", fname_scatter_svg);
            println!("===============================================");
        }
        Ok(final_cost_value)
    }

    pub fn hyperoptimise(
        self: &Self,
        selection_hidden_layers: &Vec<usize>,
        selection_hidden_layer_nodes: &Vec<usize>,
        selection_dropout_rates: &Vec<f32>,
        selection_learning_rates: &Vec<f32>,
        selection_n_epochs: &Vec<usize>,
        selection_n_burnin_epochs: &Vec<usize>,
        selection_f_patient_epochs: &Vec<f32>,
        selection_f_validation: &Vec<f32>,
        selection_n_batches: &Vec<usize>,
        selection_activations: &Vec<Activation>,
        selection_costs: &Vec<Cost>,
        selection_optimisers: &Vec<Optimiser>,
        selection_weights_initialisations: &Vec<WeightsInitialisation>,
        verbose: bool,
    ) -> Result<Self, Box<dyn Error>> {
        self.check_dimensions()?;
        let param_combinations = prep_all_hyperparams(
            selection_hidden_layers,
            selection_hidden_layer_nodes,
            selection_dropout_rates,
            selection_learning_rates,
            selection_n_epochs,
            selection_n_burnin_epochs,
            selection_f_patient_epochs,
            selection_f_validation,
            selection_n_batches,
            selection_activations,
            selection_costs,
            selection_optimisers,
            selection_weights_initialisations,
        )?;
        // Hyper-parameter optimisations
        let mut results: Vec<(
            usize,
            usize,
            f32,
            f32,
            usize,
            usize,
            f32,
            f32,
            usize,
            Activation,
            Cost,
            Optimiser,
            WeightsInitialisation,
            f32,
        )> = Vec::new();
        let mut best_params = (f32::MAX, param_combinations[0].clone());
        if verbose {
            println!(
                "Hyperparameter optimisation ({} hyperparameter combinations to test):",
                &param_combinations.len()
            );
        }
        for p in &param_combinations {
            let (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                n_burnin_epochs,
                f_patient_epochs,
                f_validation,
                n_batches,
                activation,
                cost,
                optimiser,
                weights_initialisation,
            ) = p.clone();
            if verbose {
                println!(
                    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n[ {} / {} ]",
                    &results.len() + 1,
                    &param_combinations.len(),
                );
                println!(
                    "| n_hidden_layers={:13} | n_hidden_nodes={:12} | dropout_rate={:12.4} | learning_rate={:13.6} | n_epochs={:6} | n_burnin_epochs={:6} | f_patient_epochs={:14} | f_validation={:14} | n_batches={:7} | activation={:?} | cost={:?} | optimiser={:?} | weights_initialisation={:?} |",
                    n_hidden_layers,
                    n_hidden_nodes,
                    dropout_rate,
                    learning_rate,
                    n_epochs,
                    n_burnin_epochs,
                    f_patient_epochs,
                    f_validation,
                    n_batches,
                    activation,
                    cost,
                    optimiser,
                    weights_initialisation,
                );
            }
            // Create a new instance of the network with the current hyperparameters
            let mut network = Network::new(
                &self.activations_per_layer[0]
                    .data
                    .context()
                    .default_stream(),
                self.activations_per_layer[0].clone(),
                self.targets.clone(),
                n_hidden_layers,
                vec![n_hidden_nodes; n_hidden_layers],
                vec![dropout_rate; n_hidden_layers],
                weights_initialisation,
                self.seed,
            )?;
            network.activation = activation.clone();
            network.cost = cost.clone();
            let mut optimisation_parameters = OptimisationParameters::new(&network)?;
            optimisation_parameters.learning_rate = learning_rate;
            optimisation_parameters.n_epochs = n_epochs;
            optimisation_parameters.n_burnin_epochs = n_burnin_epochs;
            optimisation_parameters.f_patient_epochs = f_patient_epochs;
            optimisation_parameters.f_validation = f_validation;
            optimisation_parameters.n_batches = n_batches;
            optimisation_parameters.optimiser = optimiser.clone();
            // Train the network with the current hyperparameters
            let loss = match network.train(&optimisation_parameters, verbose) {
                Ok(x) => x,
                Err(_) => f32::MAX,
            };
            // Check if loss is better
            if loss < best_params.0 {
                best_params = (loss, p.clone());
            }
            // Store the result of the training
            results.push((
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                n_burnin_epochs,
                f_patient_epochs,
                f_validation,
                n_batches,
                activation.clone(),
                cost.clone(),
                optimiser.clone(),
                weights_initialisation.clone(),
                loss,
            ));
        }
        // Print the results
        if verbose {
            println!("Hyper-parameter Optimisation Results:");
            println!(
                "| Hidden_Layers | Hidden_Nodes | Dropout_Rate | Learning_Rate | Epochs | Patient_Epochs | Validation_Set | Batches | Activation | Cost | Optimiser | Weights_Initialisation | Final_Cost |"
            );
            for (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                n_burnin_epochs,
                f_patient_epochs,
                f_validation,
                n_batches,
                activation,
                cost,
                optimiser,
                weights_initialisation,
                loss,
            ) in &results
            {
                println!(
                    "| {:13} | {:12} | {:12.4} | {:13.6} | {:6} | {:6} | {:14} | {:14} | {:7} | {:?} | {:?} | {:?} | {:?} | {:10.6} |",
                    n_hidden_layers,
                    n_hidden_nodes,
                    dropout_rate,
                    learning_rate,
                    n_epochs,
                    n_burnin_epochs,
                    f_patient_epochs,
                    f_validation,
                    n_batches,
                    activation,
                    cost,
                    optimiser,
                    weights_initialisation,
                    loss,
                );
            }
        }
        // Build and train the network using the best hyperparameters
        let (
            loss_expected,
            (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                n_burnin_epochs,
                f_patient_epochs,
                f_validation,
                n_batches,
                activation,
                cost,
                optimiser,
                weights_initialisation,
            ),
        ) = best_params;
        if verbose {
            println!("Best hyperparameters found:");
            println!("\t- Hidden Layers: {}", n_hidden_layers);
            println!("\t- Hidden Nodes: {}", n_hidden_nodes);
            println!("\t- Dropout Rate: {}", dropout_rate);
            println!("\t- Learning Rate: {}", learning_rate);
            println!("\t- Epochs: {}", n_epochs);
            println!("\t- Burnin Epochs: {}", n_burnin_epochs);
            println!(
                "\t- Patient Epochs: {}",
                (f_patient_epochs * n_epochs as f32).floor() as usize
            );
            println!(
                "\t- Validation Set: {}",
                (f_validation * self.targets.n_cols as f32).floor() as usize
            );
            println!("\t- Batches: {}", n_batches);
            println!("\t- Activation: {:?}", activation);
            println!("\t- Cost: {:?}", cost);
            println!("\t- Optimiser: {:?}", optimiser);
            println!("\t- Weights Initialisation: {:?}", weights_initialisation);
            println!("\t- Mean Loss: {}", loss_expected);
        }
        let mut network = Network::new(
            &self.activations_per_layer[0]
                .data
                .context()
                .default_stream(),
            self.activations_per_layer[0].clone(),
            self.targets.clone(),
            n_hidden_layers,
            vec![n_hidden_nodes; n_hidden_layers],
            vec![dropout_rate; n_hidden_layers],
            weights_initialisation,
            self.seed,
        )?;
        network.activation = activation.clone();
        network.cost = cost.clone();
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        optimisation_parameters.learning_rate = learning_rate;
        optimisation_parameters.n_epochs = n_epochs;
        optimisation_parameters.n_burnin_epochs = n_burnin_epochs;
        optimisation_parameters.f_patient_epochs = f_patient_epochs;
        optimisation_parameters.f_validation = f_validation;
        optimisation_parameters.n_batches = n_batches;
        optimisation_parameters.optimiser = optimiser.clone();
        // Train the network using the best hyperparameters
        let loss = network.train(&optimisation_parameters, verbose)?;
        if verbose {
            println!(
                "Expected loss = {} | Observed loss = {}",
                loss_expected, loss
            );
        }
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Data;
    use crate::network::WeightsInitialisation;
    #[test]
    fn test_train() -> Result<(), Box<dyn Error>> {
        let n: usize = 12_345; // number of observations
        let p: usize = 17; // number of continuous input features
        let q: Vec<usize> = vec![2,3,4,5]; // number of levels for each categorical feature variable
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let (data, _network_simulated) = Data::simulate(n, p, q, k, n_hidden_layers, "normal", 0.0, 1.0, 123, true)?;
        let mut network = data.init_network(2, vec![5; 2], vec![0.0; 2], WeightsInitialisation::He, 123)?;
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        println!("Network:\n{}\n\n", network);
        println!("Optimisation Parameters:\n{}\n\n", optimisation_parameters);
        // optimisation_parameters.learning_rate = 0.00001f32;
        // optimisation_parameters.optimiser = Optimiser::GradientDescent;
        optimisation_parameters.optimiser = Optimiser::Adam;
        // optimisation_parameters.optimiser = Optimiser::AdamMax;
        // Test shufflesplit
        let indexes: Vec<Vec<usize>> = network.shufflesplit(5)?;
        // println!("indexes: {:?}", indexes);
        println!("Number of batches: {:?}", indexes.len());
        let mut total_len: usize = 0;
        for i in 0..indexes.len() {
            println!(
                "indexes[{}]: [{}, {}, ...{}] length: {:?}",
                i,
                indexes[i][0],
                indexes[i][1],
                indexes[i][indexes[i].len() - 1],
                indexes[i].len()
            );
            total_len += indexes[i].len();
        }
        println!("Total length: {:?}", total_len);
        assert!(total_len == network.targets.n_cols);
        // Test train_per_batch
        let cost_prior_to_training: f32 = network.loss()?;
        println!("cost prior to training = {}", cost_prior_to_training);
        println!("predictions before training: {}", network.targets);
        for _ in 0..7 {
            network.train_per_batch(&mut optimisation_parameters, "1", false)?;
        }
        println!("cost after training = {}", network.loss()?);
        println!("predictions after training: {}", network.targets);
        assert!(cost_prior_to_training > network.loss()?);

        let mut network_epochs_5 = network.clone();
        let mut network_epochs_200 = network.clone();
        optimisation_parameters.n_batches = 1;
        optimisation_parameters.n_epochs = 5;
        network_epochs_5.train(&mut optimisation_parameters, false)?;
        optimisation_parameters.n_epochs = 200;
        network_epochs_200.train(&mut optimisation_parameters, false)?;
        println!("cost after training for 5 epochs = {}", network_epochs_5.loss()?);
        println!("cost after training for 200 epochs = {}", network_epochs_200.loss()?);
        assert!(network_epochs_5.loss()? > network_epochs_200.loss()?);

        // Hyper-parameter optimisations
        let selection_hidden_layers = vec![1, 2];
        let selection_hidden_layer_nodes = vec![5];
            let selection_dropout_rates = vec![0.0];
        let selection_learning_rates = vec![0.0001];
        let selection_n_epochs = vec![1, 2, 3];
        let selection_n_burnin_epochs = vec![0, 1, 2];
        let selection_f_patient_epochs = vec![0.5];
        let selection_f_validation = vec![0.0, 0.1];
        let selection_n_batches = vec![1, 2];
        let selection_activations = vec![Activation::ReLU];
        let selection_costs = vec![Cost::MSE];
        let selection_optimisers = vec![Optimiser::GradientDescent];
        let selection_weights_initialisations = vec![WeightsInitialisation::He, WeightsInitialisation::Cauchy];
        
        let verbose = false;
        let network_hyper_optimised = network.hyperoptimise(
            &selection_hidden_layers,
            &selection_hidden_layer_nodes,
            &selection_dropout_rates,
            &selection_learning_rates,
            &selection_n_epochs,
            &selection_n_burnin_epochs,
            &selection_f_patient_epochs,
            &selection_f_validation,
            &selection_n_batches,
            &selection_activations,
            &selection_costs,
            &selection_optimisers,
            &selection_weights_initialisations,
            verbose,
        )?;
        println!("network_hyper_optimised:\n{}", network_hyper_optimised);
        // Clean-up
        for f in std::fs::read_dir(".")? {
            let f = f?.path();
            if f.is_file() && (f.extension().and_then(|s| s.to_str()) == Some("png") || f.extension().and_then(|s| s.to_str()) == Some("svg")) {
                std::fs::remove_file(&f)?;
            }
        }
        Ok(())
    }
}
