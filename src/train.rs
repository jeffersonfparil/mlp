use crate::activations::Activation;
use crate::costs::Cost;
use crate::network::Network;
use crate::optimisers::{OptimisationParameters, Optimiser};
use chrono::Utc;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;
use std::env::current_dir;
use std::error::Error;
use std::fmt;
use std::ops::Add;
use std::path::PathBuf;
use std::sync::Mutex;
// use cudarc::driver::CudaSlice;
// use crate::linalg::matrix::Matrix;
use std::cmp::Ordering;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Debug, PartialEq)]
enum TrainingError {
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

impl From<PlottingError> for TrainingError {
    fn from(err: PlottingError) -> Self {
        TrainingError::OtherError(err.to_string())
    }
}

fn prep_each_hyperparam<T>(param_min_max_step: (T, T, T)) -> Result<Vec<T>, Box<dyn Error>>
where
    T: Copy + PartialOrd + Add<Output = T> + Default + PartialEq,
{
    let (min, max, step) = param_min_max_step;
    if step == T::default() {
        return Err(Box::new(TrainingError::OtherError(
            "Step must be non-zero.".to_string(),
        )));
    }
    // if step > 0 and max < min -> invalid; if step < 0 and max > min -> invalid
    if (step > T::default() && max < min) || (step < T::default() && max > min) {
        return Err(Box::new(TrainingError::OtherError(
            "Invalid range.".to_string(),
        )));
    }
    let mut selection: Vec<T> = Vec::new();
    let mut x = min;
    if step > T::default() {
        while x <= max {
            selection.push(x);
            x = x + step;
        }
    } else {
        while x >= max {
            selection.push(x);
            x = x + step; // step is negative here
        }
    }
    Ok(selection)
}

fn prep_all_hyperparams(
    range_hidden_layers: Option<(usize, usize, usize)>,
    range_hidden_layer_nodes: Option<(usize, usize, usize)>,
    range_dropout_rate: Option<(f32, f32, f32)>,
    range_learning_rate: Option<(f32, f32, f32)>,
    range_n_epochs: Option<(usize, usize, usize)>,
    range_f_patient_epochs: Option<(f32, f32, f32)>,
    range_n_batches: Option<(usize, usize, usize)>,
    selection_activations: Option<Vec<Activation>>,
    selection_costs: Option<Vec<Cost>>,
    selection_optimisers: Option<Vec<Optimiser>>,
) -> Result<
    Vec<(
        usize,
        usize,
        f32,
        f32,
        usize,
        f32,
        usize,
        Activation,
        Cost,
        Optimiser,
    )>,
    Box<dyn Error>,
> {
    let selection_hidden_layers: Vec<usize> = match range_hidden_layers {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1, 3, 1))?,
    };
    let selection_hidden_layer_nodes: Vec<usize> = match range_hidden_layer_nodes {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((100, 500, 100))?,
    };
    let selection_dropout_rates: Vec<f32> = match range_dropout_rate {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((0.0, 0.5, 0.01))?,
    };
    let selection_learning_rates: Vec<f32> = match range_learning_rate {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1e-5, 1e-2, 1e-4))?,
    };
    let selection_n_epochs: Vec<usize> = match range_n_epochs {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((5, 10, 1))?,
    };
    let selection_f_patient_epochs: Vec<f32> = match range_f_patient_epochs {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((0.5, 1.0, 0.5))?,
    };
    let selection_n_batches: Vec<usize> = match range_n_batches {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1, 3, 1))?,
    };
    let selection_activations: Vec<Activation> = match selection_activations {
        Some(x) => x,
        None => vec![Activation::ReLU],
    };
    let selection_costs: Vec<Cost> = match selection_costs {
        Some(x) => x,
        None => vec![Cost::MSE],
    };
    let selection_optimisers: Vec<Optimiser> = match selection_optimisers {
        Some(x) => x,
        None => vec![
            Optimiser::Adam,
            Optimiser::AdamMax,
            Optimiser::GradientDescent,
        ],
    };
    let mut param_combinations: Vec<(
        usize,
        usize,
        f32,
        f32,
        usize,
        f32,
        usize,
        Activation,
        Cost,
        Optimiser,
    )> = Vec::new();
    for n_hidden_layers in &selection_hidden_layers {
        for n_hidden_nodes in &selection_hidden_layer_nodes {
            for dropout_rate in &selection_dropout_rates {
                for learning_rate in &selection_learning_rates {
                    for n_epochs in &selection_n_epochs {
                        for f_patient_epochs in &selection_f_patient_epochs {
                            for n_batches in &selection_n_batches {
                                for activation in &selection_activations {
                                    for cost in &selection_costs {
                                        for optimiser in &selection_optimisers {
                                            param_combinations.push((
                                                *n_hidden_layers,
                                                *n_hidden_nodes,
                                                *dropout_rate,
                                                *learning_rate,
                                                *n_epochs,
                                                *f_patient_epochs,
                                                *n_batches,
                                                activation.clone(),
                                                cost.clone(),
                                                optimiser.clone(),
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
        batch_id: &str,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let mut epochs: Vec<f64> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();
        let n_patient_epochs = (optimisation_parameters.f_patient_epochs
            * optimisation_parameters.n_epochs as f32)
            .ceil() as usize;
        // ///////////////////////////////////////////////////////////////////////////
        // ///////////////////////////////////////////////////////////////////////////
        // ///////////////////////////////////////////////////////////////////////////
        // // With cross-validation
        // const FRAC_VALIDATION: f32 = 0.05;
        // let n: usize = self.targets.n_cols;
        // let n_validation: usize = if (n as f32 * FRAC_VALIDATION).floor() < 1.0 {
        //     3
        // } else {
        //     (n as f32 * FRAC_VALIDATION).floor() as usize
        // };
        // let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        // let validation_indexes: Vec<usize> = (0..n).choose_multiple(&mut rng, n_validation);
        // let training_indexes: Vec<usize> = (0..n)
        //     .filter(|&x| !validation_indexes.contains(&x))
        //     .collect();
        // let mut network_validation = self.slice(&validation_indexes)?;
        // let mut network_training = self.slice(&training_indexes)?;
        // for epoch in 0..optimisation_parameters.n_epochs {
        //     network_training.forwardpass()?;
        //     network_training.backpropagation()?;
        //     network_training.optimise(optimisation_parameters)?;
        //     network_training.predict()?;
        //     epochs.push(epoch as f64);
        //     // Validate
        //     network_validation.replace_model(&network_training)?;
        //     network_validation.predict()?;
        //     costs.push(network_validation.loss()? as f64);
        //     // Update the network after training the training network
        //     self.replace_model(&network_training)?;
        //     // Early stopping check, i.e. stop if no improvement in cost after n_patient_epochs
        //     if (epoch > n_patient_epochs) && (costs[epoch] >= costs[epoch - n_patient_epochs]) {
        //         // println!("Early stopping at epoch {}", epoch);
        //         break;
        //     }
        // }
        // ///////////////////////////////////////////////////////////////////////////
        // ///////////////////////////////////////////////////////////////////////////
        // ///////////////////////////////////////////////////////////////////////////
        // No cross-validation
        let start_time = Instant::now();
        for epoch in 0..optimisation_parameters.n_epochs {
            let perc: f64 = (1_00_00.0 * (epoch as f64 + 1.00)/(optimisation_parameters.n_epochs as f64)).round() / 100.0;
            let n_progress: usize = (100.0 * ((epoch+1) as f64) / (optimisation_parameters.n_epochs as f64)).round() as usize;
            let progress_text: String = (0..n_progress).map(|_| "█").collect();
            let no_progress_text: String = (0..(100-n_progress)).map(|_| " ").collect();
            print!("\rTraining batch {} | {:.2}% | {}{} |", batch_id, perc, progress_text, no_progress_text);
            io::stdout().flush().expect("Failed to flush stdout");
            self.forwardpass()?;
            self.backpropagation()?;
            self.optimise(optimisation_parameters)?;
            self.predict()?;
            epochs.push(epoch as f64);
            costs.push(self.loss()? as f64);
            // Early stopping check, i.e. stop if no improvement in cost after n_patient_epochs
            if (epoch > n_patient_epochs) && (costs[epoch] >= costs[epoch - n_patient_epochs]) {
                // println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
        println!(" Duration: {:.2} minutes", start_time.elapsed().as_millis() as f64 / 60_000.0);
        self.predict()?;
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
        let (epochs, costs): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
            if optimisation_parameters.n_batches == 1 {
                // Only one batch, train on the whole dataset
                let mut params = optimisation_parameters.clone();
                let (epochs, costs) = self.train_per_batch(&mut params, "1/1")?;
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
                let epochs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
                let costs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
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
                        let result = network.train_per_batch(&mut params, &format!("{} / {}", i+1, optimisation_parameters.n_batches));
                        match result {
                            Ok((epochs_batch, costs_batch)) => {
                                epochs.lock().unwrap().push(epochs_batch);
                                costs.lock().unwrap().push(costs_batch);
                            }
                            Err(e) => {
                                // Skip the batch
                                eprintln!("Error training on batch {}: {}", i+1, e);
                            }
                        }
                    });
                // Merge the parameters from each batch network back into the original network via simple averaging with a better method
                self.average_weights_biases(&networks_per_batch)?;
                // Return epochs, costs
                (epochs.into_inner().unwrap(), costs.into_inner().unwrap())
            };
        // Assess cost after training
        let final_cost_value = self.loss()?;
        if verbose {
            // Plot loss curve
            let dir: PathBuf = current_dir()?;
            let fname_loss_svg = &format!(
                "{}/Loss_curve-HL{}-{:?}-{:?}-E{}-FPE{}-B{}-LR{}-T{}.svg",
                dir.display(),
                self.n_hidden_layers,
                self.activation,
                optimisation_parameters.optimiser,
                optimisation_parameters.n_epochs,
                optimisation_parameters.f_patient_epochs,
                optimisation_parameters.n_batches,
                optimisation_parameters.learning_rate,
                Utc::now().format("%Y%m%d%H%M%S")
            );
            let fname_scatter_svg = &fname_loss_svg.replace("Loss_curve-", "Observed_vs_predicted-");
            let mut ylabel = String::from("Cost");
            ylabel.push_str(&format!(
                " ({:?}; {:?})",
                self.cost, optimisation_parameters.optimiser
            ));
            let mut plot_loss = vec![
                Plot::new()
                    .title("Training Cost over Epochs")
                    .legend_position(LegendPosition::Best)
                    .xlabel("Epochs")
                    .ylabel(&ylabel)
                    .line(&epochs[0], &costs[0])
                    .label("Batch 0")
                    .size(4.0, 3.0),                
            ];
            for i in 1..optimisation_parameters.n_batches {
                plot_loss[0] = plot_loss[0]
                    .clone()
                    .line(&epochs[i], &costs[i])
                    .label(&format!("Batch {}", i+1));
            }
            ;
            // plot_loss[0].clone().save(fname_png)?;
            plot_loss[0].clone().export_svg(fname_loss_svg)?;
            // Scatter plot of observed vs predicted values
            let y_observed: Vec<f64> = self.targets.to_host()?.iter().map(|&x| x as f64).collect();
            let y_predicted: Vec<f64> = self.predictions.to_host()?.iter().map(|&x| x as f64).collect();
            // OLS
            let epsilon: f64 = 1e-7;
            let n: f64 = y_observed.len() as f64;
            let u_observed: f64 = y_observed.iter().fold(0.0, |sum, x| sum + x) / n;
            let u_predicted: f64 = y_predicted.iter().fold(0.0, |sum, x| sum + x) / n;
            let cov_xy: f64 = y_observed
                .iter()
                .zip(y_predicted.iter())
                .fold(0.0, |a, (x, y)| a + (x - u_observed) * (y - u_predicted));
            let var_x: f64 = y_observed
                .iter()
                .fold(0.0, |a, x| a + (x - u_observed).powi(2));
            let var_y: f64 = y_predicted
                .iter()
                .fold(0.0, |a, x| a + (x - u_predicted).powi(2));
            let b: f64 = cov_xy / (var_x + epsilon);
            let a: f64 = u_predicted - (b * u_observed);
            let error2: f64 = y_predicted
                .iter()
                .zip(y_observed.iter())
                .fold(0.0, |a, (y_p, y_t)| a + (y_p - y_t).powi(2));
            let mse: f64 = error2 / n;
            let r2: f64 = 1.00 - (error2 / (var_x + epsilon));
            let cor: f64 = cov_xy / ((var_x.sqrt() * var_y.sqrt()) + epsilon);
            let x_min: f64 = match y_observed
                .iter()
                .min_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Less)
                }) {
                    Some(x) => *x,
                    None => y_observed[0],
                };
            let x_max: f64 = match y_observed
                .iter()
                .max_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(Ordering::Greater)
                }) {
                    Some(x) => *x,
                    None => y_observed[0],
                };
            let mut x_for_plotting: Vec<f64> = Vec::with_capacity(100);
            let mut y_for_plotting: Vec<f64> = Vec::with_capacity(100);
            let step: f64 = (x_max - x_min) / 99.0;
            for i in 0..100 {
                let x: f64 = x_min + (i as f64 * step);
                let y: f64 = a + x*b;
                x_for_plotting.push(x);
                y_for_plotting.push(y);
            }
            let title = format!("Observed vs Predicted Values\n(n={}; MSE={:.2}; R2={:.2}; cor={:.2})", n, mse, r2, cor);
            Plot::new()
                .title(title.as_str())
                .xlabel("Observed")
                .ylabel("Predicted")
                .scatter(&y_observed, &y_predicted)
                .line(&x_for_plotting, &y_for_plotting)
                .export_svg(fname_scatter_svg)?;

            // Messages
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
        range_hidden_layers: Option<(usize, usize, usize)>,
        range_hidden_layer_nodes: Option<(usize, usize, usize)>,
        range_dropout_rate: Option<(f32, f32, f32)>,
        range_learning_rate: Option<(f32, f32, f32)>,
        range_n_epochs: Option<(usize, usize, usize)>,
        range_f_patient_epochs: Option<(f32, f32, f32)>,
        range_n_batches: Option<(usize, usize, usize)>,
        selection_activations: Option<Vec<Activation>>,
        selection_costs: Option<Vec<Cost>>,
        selection_optimisers: Option<Vec<Optimiser>>,
        verbose: bool,
    ) -> Result<Self, Box<dyn Error>> {
        self.check_dimensions()?;
        let param_combinations = prep_all_hyperparams(
            range_hidden_layers,
            range_hidden_layer_nodes,
            range_dropout_rate,
            range_learning_rate,
            range_n_epochs,
            range_f_patient_epochs,
            range_n_batches,
            selection_activations,
            selection_costs,
            selection_optimisers,
        )?;
        // Hyper-parameter optimisations
        let mut results: Vec<(
            usize,
            usize,
            f32,
            f32,
            usize,
            f32,
            usize,
            Activation,
            Cost,
            Optimiser,
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
            if verbose {
                println!(
                    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n[ {} / {} ]",
                    &results.len() + 1,
                    &param_combinations.len(),
                );
            }
            let (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                f_patient_epochs,
                n_batches,
                activation,
                cost,
                optimiser,
            ) = p.clone();
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
                self.seed,
            )?;
            network.activation = activation.clone();
            network.cost = cost.clone();
            let mut optimisation_parameters = OptimisationParameters::new(&network)?;
            optimisation_parameters.learning_rate = learning_rate;
            optimisation_parameters.n_epochs = n_epochs;
            optimisation_parameters.f_patient_epochs = f_patient_epochs;
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
                f_patient_epochs,
                n_batches,
                activation.clone(),
                cost.clone(),
                optimiser.clone(),
                loss,
            ));
        }
        // Print the results
        if verbose {
            println!("Hyper-parameter Optimisation Results:");
            println!(
                "| Hidden_Layers | Hidden_Nodes | Dropout_Rate | Learning_Rate | Epochs | Patient_Epochs | Batches | Activation | Cost | Optimiser | Final_Cost |"
            );
            for (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                f_patient_epochs,
                n_batches,
                activation,
                cost,
                optimiser,
                loss,
            ) in &results
            {
                println!(
                    "| {:13} | {:12} | {:12.4} | {:13.6} | {:6} | {:14} | {:7} | {:?} | {:?} | {:?} | {:10.6} |",
                    n_hidden_layers,
                    n_hidden_nodes,
                    dropout_rate,
                    learning_rate,
                    n_epochs,
                    f_patient_epochs,
                    n_batches,
                    activation,
                    cost,
                    optimiser,
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
                f_patient_epochs,
                n_batches,
                activation,
                cost,
                optimiser,
            ),
        ) = best_params;
        if verbose {
            println!("Best parameter found:");
            println!("\t- Hidden Layers: {}", n_hidden_layers);
            println!("\t- Hidden Nodes: {}", n_hidden_nodes);
            println!("\t- Dropout Rate: {}", dropout_rate);
            println!("\t- Learning Rate: {}", learning_rate);
            println!("\t- Epochs: {}", n_epochs);
            println!(
                "\t- Patient Epochs: {}",
                (f_patient_epochs * n_epochs as f32).ceil() as usize
            );
            println!("\t- Batches: {}", n_batches);
            println!("\t- Activation: {:?}", activation);
            println!("\t- Cost: {:?}", cost);
            println!("\t- Optimiser: {:?}", optimiser);
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
            self.seed,
        )?;
        network.activation = activation.clone();
        network.cost = cost.clone();
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        optimisation_parameters.learning_rate = learning_rate;
        optimisation_parameters.n_epochs = n_epochs;
        optimisation_parameters.f_patient_epochs = f_patient_epochs;
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
    use cudarc::driver::{CudaContext, CudaSlice};
    #[test]
    fn test_train() -> Result<(), Box<dyn Error>> {
        let n: usize = 12_345; // number of observations
        let p: usize = 17; // number of input features
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let data = Data::simulate(n, p, k, n_hidden_layers, "normal", 0.0, 1.0, 42)?;
        let mut network = data.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
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
        network.train_per_batch(&mut optimisation_parameters, "1")?;
        println!("cost after training = {}", network.loss()?);
        println!("predictions after training: {}", network.targets);
        assert!(cost_prior_to_training > network.loss()?);

        let mut network_epochs_5 = network.clone();
        let mut network_epochs_200 = network.clone();
        optimisation_parameters.n_batches = 1;
        optimisation_parameters.n_epochs = 5;
        network_epochs_5.train(&mut optimisation_parameters, true)?;
        optimisation_parameters.n_epochs = 200;
        network_epochs_200.train(&mut optimisation_parameters, true)?;
        println!("cost after training for 5 epochs = {}", network_epochs_5.loss()?);
        println!("cost after training for 200 epochs = {}", network_epochs_200.loss()?);
        assert!(network_epochs_5.loss()? > network_epochs_200.loss()?);

        // Hyper-parameter optimisations
        let range_hidden_layers = Some((1, 2, 1));
        let range_hidden_layer_nodes = Some((5, 5, 5));
        let range_dropout_rate = Some((0.0, 0.0, 0.1));
        let range_learning_rate = Some((0.0001, 0.0001, 0.0001));
        let range_n_epochs = Some((1, 3, 1));
        let range_f_patient_epochs = Some((0.5, 0.5, 0.5));
        let range_n_batches = Some((1, 2, 1));
        let selection_activations = Some(vec![Activation::ReLU]);
        let selection_costs = Some(vec![Cost::MSE]);
        let selection_optimisers = Some(vec![Optimiser::Adam, Optimiser::GradientDescent]);
        let verbose = true;
        let network_hyper_optimised = network.hyperoptimise(
            range_hidden_layers,
            range_hidden_layer_nodes,
            range_dropout_rate,
            range_learning_rate,
            range_n_epochs,
            range_f_patient_epochs,
            range_n_batches,
            selection_activations,
            selection_costs,
            selection_optimisers,
            verbose,
        )?;
        println!("network_hyper_optimised:\n{}", network_hyper_optimised);
        // Clean-up
        for f in std::fs::read_dir(".")? {
            let f = f?.path();
            if f.is_file() && f.extension().and_then(|s| s.to_str()) == Some("svg") {
                std::fs::remove_file(&f)?;
            }
        }
        Ok(())
    }
}
