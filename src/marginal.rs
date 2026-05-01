use crate::linalg::matrix::Matrix;
use crate::network::Network;
use crate::progress_bar::ProgressBar;
use std::error::Error;
use itertools::Itertools;
use std::fmt;
use rayon::prelude::*;
use std::sync::Mutex;
use std::sync::Arc;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum MarginalError {
    DimensionMismatch(String),
    NameMismatch(String),
    OtherError(String),
}

impl Error for MarginalError {}

impl fmt::Display for MarginalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MarginalError::DimensionMismatch(msg) => {
                write!(f, "Dimension Mismatch in Marginals: {}", msg)
            }
            MarginalError::NameMismatch(msg) => {
                write!(f, "Name Mismatch in Marginals: {}", msg)
            }
            MarginalError::OtherError(msg) => write!(f, "Other Error in Marginals: {}", msg),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Marginals {
    pub ids: Vec<String>,
    pub effects: Vec<f32>,
    pub r2s: Vec<f32>,
}

impl Marginals {
    pub fn new(feature_names: Vec<String>, order: usize) -> Result<Self, Box<dyn Error>> {
        let n = feature_names.len();
        let mut ids: Vec<String> = vec![];
        for i in 1..=order {
            'combi: for combination in (0..n).into_iter().combinations(i) {
                // Skip we have duplicate features in a combination (which we should not have because `combinations(i)` yield all possible i-combinations where order does not matter), and 
                //      if we have non-numeric features if they are the same feature just different levels
                let m = combination.len();
                if m > 1 {
                    for j in 0..(m-1) {
                        let c0 = combination[j].to_owned();
                        let f0 = match feature_names[c0].to_owned().split("➵").next() {
                            Some(x) => x.to_owned(),
                            None => feature_names[c0].to_owned(),
                        };
                        for k in (j+1)..m {
                            let c1 = combination[k].to_owned();
                            let f1 = match feature_names[c1].to_owned().split("➵").next() {
                                Some(x) => x.to_owned(),
                                None => feature_names[c1].to_owned(),
                            };
                            if c0 == c1 {
                                continue 'combi; 
                            }
                            if f0 == f1 {
                                continue 'combi; 
                            }
                        }
                    }
                }
                for (j, k) in combination.into_iter().enumerate() {
                    if j == 0 {
                        ids.push(feature_names[k].to_owned())
                    } else {
                        let idx = ids.len()-1;
                        ids[idx] = format!("{}▓{}", ids[ids.len()-1], feature_names[k])
                    };
                    // println!("ids[idx]={}", ids[idx]);
                }
            }
        }
        let p = ids.len();
        let marginals  = Marginals {
            ids,
            effects: vec![f32::NAN; p],
            r2s: vec![f32::NAN; p],
        };
        Ok(marginals)
    }

    pub fn check_dimensions(&self) -> Result<(), MarginalError> {
        if self.ids.len() != self.effects.len() {
            return Err(MarginalError::DimensionMismatch(format!(
                "Number of ids ({}) does not match number of effects ({}).",
                self.ids.len(), self.effects.len()
            )));
        }
        if self.ids.len() != self.r2s.len() {
            return Err(MarginalError::DimensionMismatch(format!(
                "Number of ids ({}) does not match number of r2s ({}).",
                self.ids.len(), self.r2s.len()
            )));
        }
        if self.effects.len() != self.r2s.len() {
            return Err(MarginalError::DimensionMismatch(format!(
                "Number of effects ({}) does not match number of r2s ({}).",
                self.effects.len(), self.r2s.len()
            )));
        }
        Ok(())
    }

    pub fn unstandaridise(self: &mut Self, network: &Network) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let mean: f32 = network.targets_mean_sd.0;
        let sd: f32 = network.targets_mean_sd.1;
        for i in 0..self.effects.len() {
            self.effects[i] = (sd*self.effects[i]) + mean;
        }
        Ok(())
    }
    
    pub fn estimate_perturb(self: &mut Self, network_orig: &Network, n_interpolate_min_max: usize, verbose: bool) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        // Find the range of values for each input node
        // println!("number of activation layers: {}", network_orig.activations_per_layer.len());
        // println!("input_layer: {}", network_orig.activations_per_layer[0]);
        let n: usize = network_orig.activations_per_layer[0].n_cols;
        let p: usize = network_orig.activations_per_layer[0].n_rows;
        // let mut minima: Vec<f32> = vec![f32::NAN; p];
        // let mut maxima: Vec<f32> = vec![f32::NAN; p];
        let input_matrix_orig = network_orig.activations_per_layer[0].to_host()?;
        // let mut input_matrix = input_matrix_orig.clone();
        // let stream = network_orig.activations_per_layer[0].data.context().default_stream();
        let mut feature_names: Vec<String> = vec![];
        for i in 0..self.ids.len() {
            let id = self.ids[i].to_owned();
            let id_split = id.split("▓").into_iter().map(|x| x.to_owned()).collect::<Vec<String>>();
            if id_split.len() == 1 {
                feature_names.push(id);
            }
        }
        // Emit custom MarginalError here
        if p != feature_names.len() {
            return Err(Box::new(MarginalError::DimensionMismatch(format!("The Network has {} features but the Marginals struct has {} features (i.e. {:?}).", p, feature_names.len(),feature_names))));
        }
        // Define the ranges for each of the features
        let mut ranges: Vec<Vec<f32>> = vec![];
        for j in 0..p {
            let ini: usize = j * n;
            let fin: usize = (j + 1) * n;
            let old_values = match input_matrix_orig.get(ini..fin) {
                Some(x) => x.to_owned(),
                None => return Err(Box::new(MarginalError::DimensionMismatch(format!("Inappropriate slicing index from {} to {}.", ini, fin)))),
            };
            // println!("old_values[0]={}, old_values[1]={}, old_values[2]={}, old_values[3]={}", old_values[0], old_values[1], old_values[2], old_values[3]);
            let min = match old_values.iter().filter(|&a| !a.is_nan()).min_by(|&a, &b| a.total_cmp(b)) {
                Some(&a) => a,
                None => f32::NAN,
            };
            let max = match old_values.iter().filter(|&a| !a.is_nan()).max_by(|&a, &b| a.total_cmp(b)) {
                Some(&a) => a,
                None => f32::NAN,
            };
            let step_size = (max - min) / ((n_interpolate_min_max-1) as f32);
            let new_values_to_iterate: Vec<f32> = (0..n_interpolate_min_max).map(|x| min+(step_size*(x as f32))).collect();
            ranges.push(new_values_to_iterate);
        }
        // println!("ranges: {:?}", ranges);
        // let start_time = Instant::now();
        // let progress_width: usize = 50;
        // let counter = Arc::new(atomic::AtomicUsize::new(0));
        let pb = Arc::new(Mutex::new(ProgressBar::new(self.ids.len(), 50, format!("Estimating {} marginal effects", self.ids.len()))));
        let effects: Mutex<Vec<f32>> = Mutex::new(vec![f32::NAN; self.ids.len()]);
        let r2s: Mutex<Vec<f32>> = Mutex::new(vec![f32::NAN; self.ids.len()]);
        self.ids
            .par_iter()
            .enumerate()
            .for_each(|(i, id)| {
        // let ids = self.ids.clone();
        // for (i, id) in ids.into_iter().enumerate() {
            if verbose {
                pb.lock().unwrap().next();
            }
            // let id = self.ids[i].to_owned();
            let id_split = id.split("▓").into_iter().map(|x| x.to_owned()).collect::<Vec<String>>();
            // Find the index of the feature name for each id which may contain a single or a combination of 2 or more feature names
            let mut idx_split: Vec<usize> = vec![];
            for j in 0..id_split.len() {
                let x = id_split[j].to_owned();
                let y = feature_names
                    .iter()
                    .enumerate()
                    .filter(move |&(_j, z)| &x == z)
                    .map(|(k, _)| k)
                    .collect::<Vec<usize>>();
                if y.len() != 1 {
                    // return Err(Box::new(MarginalError::NameMismatch(format!("Unrecognised feature name: `{}`", id_split[j].to_owned()))))
                    eprintln!("Unrecognised feature name: `{}`", id_split[j].to_owned());
                }
                idx_split.push(y[0]);
            }
            ////////////////////////////////////////
            ////////////////////////////////////////
            // Estimate marginal effects including the main effects and higher-order interaction effects (if any)
            //  where we explore various combinations of the interacting features via random sampling across the pre-defined range of each feature
            ////////////////////////////////////////
            ////////////////////////////////////////
            let mut x: Vec<f64> = vec![f64::NAN; n_interpolate_min_max*n]; // new input values
            let mut y: Vec<f64> = vec![f64::NAN; n_interpolate_min_max*n]; // resulting changes to predictions
            let mut network = network_orig.clone();
            let mut input_matrix = input_matrix_orig.clone();
            let stream = network.activations_per_layer[0].data.context().default_stream();
            // For each value in the new x-range we predict
            for j in 0..n_interpolate_min_max {
                // First we need to define the new x-values for all the features, where
                // for each feature index in the current combination
                for idx in idx_split.clone() {
                    // Get the current value from the predefined range for this feature, if we are dealing with a main effect, otherwise
                    // we randomly sample from the range of values so that we explore various combinations of the interacting features (not just along the same direction)
                    let x_j = if idx_split.len() == 1 {
                        ranges[idx][j]
                    } else {
                        let mut rng = ChaCha12Rng::seed_from_u64(j as u64); // using j as a randomisation seed
                        let k: usize = rng.random_range(0..n_interpolate_min_max);
                        ranges[idx][k]
                    };
                    // Calculate the starting index in the flattened input matrix
                    let ini: usize = idx * n;
                    // Update all observations for this feature
                    for k in 0..n {
                        // Set the input matrix value to the current range value
                        input_matrix[ini+k] = x_j;
                        // Track the x values: initialize if NaN, otherwise multiply for interaction effects
                        x[(j*n)+k] = if x[(j*n)+k].is_nan() {
                            x_j as f64
                        } else {
                            x[(j*n)+k] * (x_j as f64)
                        };
                    }
                }
                // Predict at the current x-values combination
                // network.activations_per_layer[0].data = stream.clone_htod(&input_matrix)?;
                network.activations_per_layer[0].data = match stream.clone_htod(&input_matrix) {
                    Ok(x) => x,
                    Err(_) => return eprintln!("Error cloning input matrix into the first layer of the activations.")
                };
                // network.predict()?;
                match network.predict() {
                    Ok(_) => (),
                    Err(_) => return eprintln!("Error in prediction.")
                };
                // let predictions = network.predictions.to_host()?;
                let predictions = match network.predictions.to_host() {
                    Ok(x) => x,
                    Err(_) => return eprintln!("Error extracting the predictions.")
                };
                for k in 0..n {
                    y[(j*n)+k] = predictions[k] as f64;
                }
                // Reset input_matrix
                for idx in idx_split.clone() {
                    let ini: usize = idx * n;
                    for k in 0..n {
                        input_matrix[ini+k] = input_matrix_orig[ini+k];
                    }
                }
            }
            // println!("x = {:?}", x);
            // println!("y = {:?}", y);
            let (b, r2): (f64, f64) = {
                let epsilon: f64 = 1e-7;
                let n: f64 = x.len() as f64;
                let u_x: f64 = x.iter().fold(0.0, |sum, x| sum + x) / n;
                let u_y: f64 = y.iter().fold(0.0, |sum, x| sum + x) / n;
                let cov_xy: f64 = x
                    .iter()
                    .zip(y.iter())
                    .fold(0.0, |a, (x, y)| a + (x - u_x) * (y - u_y));
                let var_x: f64 = x
                    .iter()
                    .fold(0.0, |a, x| a + (x - u_x).powi(2));
                let b = cov_xy / (var_x + epsilon);
                let a: f64 = u_y - (b * u_x);
                let y_hat: Vec<f64> = x.iter().map(|&x_i| a + (x_i*b)).collect();
                let sse: f64 = y_hat
                    .iter()
                    .zip(y.iter())
                    .fold(0.0, |a, (y_p, y_t)| a + (y_p - y_t).powi(2));
                let r2: f64 = 1.00 - (sse / (var_x + epsilon));
                (b, r2)
            };
            // self.effects[i] = b as f32;
            // Lock the mutexes to get access
            let mut locked_effects = effects.lock().unwrap();
            let mut locked_r2s = r2s.lock().unwrap();
            // Replace the ith element safely
            if let Some(x) = locked_effects.get_mut(i) {
                *x = b as f32;
            }
            if let Some(x) = locked_r2s.get_mut(i) {
                *x = r2 as f32;
            }
            // println!("Higher-degree effects: {} = {}", self.ids[i], self.effects[i]);
            // // Reset the network to previous state
            // network.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
            // network.predict()?;
        });
        // }
        self.effects = effects.into_inner().unwrap();
        self.r2s = r2s.into_inner().unwrap();
        // Unstandardise the effects to be more straightforward to interpret
        self.unstandaridise(network_orig)?;
        if verbose {
            // let progress_text: String = (0..progress_width).map(|_| "█").collect();
            // print!("\rEstimating {} marginal effects | 100.00% | {} |", self.ids.len(), progress_text);
            // io::stdout().flush().expect("Failed to flush stdout");
            // println!(" Duration: {:.2} minutes", start_time.elapsed().as_millis() as f64 / 60_000.0);
            pb.lock().unwrap().finish();
            let fname_marginals_effects_png: String = self.plot(true)?; // Main effects only as the interaction effects may explode and we leave it to the users to plot them using some other plotting software
            println!("===============================================");
            println!("Find the marginal effects (perturbation estimates) barplot saved as: {}", fname_marginals_effects_png);
            println!("===============================================");
        }
        // // Reset the network to previous state
        // network.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
        // network.predict()?;
        Ok(())
    }

    pub fn estimate_deepshap(self: &mut Self, network: &mut Network, r: usize, seed: usize, verbose: bool) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let n: usize = network.activations_per_layer[0].n_cols;
        let p: usize = network.activations_per_layer[0].n_rows;
        let stream = network.activations_per_layer[0].data.context().default_stream();
        // Generate SHAP values for r replications of random samnpling across the distributions of the observed input features (assuming normal distribution of these features)
        let mut shaps: Vec<Matrix> = Vec::with_capacity(r);
        let mut pb = ProgressBar::new(r, 50, format!("DeepSHAP estimation for {} replications", r));
        for _ in 0..r {
            // Using the properties of the features including the fact that some are continuous while others are categorical
            let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
            let input_reference_1_host: Vec<f32> = network.activations_per_layer[0].to_host()?;
            let mut input_reference_2_host: Vec<f32> = Vec::with_capacity(p*n);
            // Initialise vectors used in the innner loop across features
            let mut row_indexes: Vec<usize> = vec![0];
            let col_indexes: Vec<usize> = (0..p).collect();
            let mut categorical_level_marker: Vec<bool> = vec![false; n];
            for i in 0..p {
                row_indexes[0] = i;
                let y: Matrix = network.activations_per_layer[0].slice(&row_indexes, &col_indexes)?;
                let y_mean: f32 = y.meanmat()?;
                let y_var: f32 = y.varmat()?;
                let y_min: f32 = y.min()?;
                let y_max: f32 = y.max()?;
                let distribution = Normal::new(y_mean as f64, (y_var as f64).sqrt())?;
                let y_hat: Vec<f32> = (&mut rng)
                    .sample_iter(distribution)
                    .take(n)
                    .map(|x| 
                        if (x as f32) < y_min {
                            y_min
                        } else if (x as f32) > y_max {
                            y_max
                        } else {
                            x as f32
                    })
                    .collect::<Vec<f32>>();
                let id: String = self.ids[i].to_owned();
                let id_split: Vec<&str> = id.split("➵").collect();
                // Determine is the current feature is the last level of a categorical variable
                let last_level_of_categorical: bool = if i == p-1 {
                    true
                } else {
                    let id_next_split: Vec<&str> = self.ids[i+1].split("➵").collect();
                    if (id_next_split.len() > 1) && (id_next_split[id_next_split.len() - 1] == "0") {
                        true
                    } else {
                        false
                    }
                };
                // Reset categorical_level_marker for every start of each categorical feature, i.e. with zero '0' as its trailing id
                if id_split.len() > 1 {
                    if id_split[id_split.len() -1] == "0" {
                        categorical_level_marker = vec![false; n];
                    }
                }
                for j in 0..n {
                    if id_split.len() == 1 {
                        // Continuous
                        input_reference_2_host.push(y_hat[j]);
                    } else {
                        // Categorical
                        let val: f32 = if categorical_level_marker[j] {
                            0.0
                        } else if last_level_of_categorical && !categorical_level_marker[j] {
                            1.00
                        } else {
                            if y_hat[j] > 0.5 {1.00} else {0.0}
                        };
                        input_reference_2_host.push(val);
                        categorical_level_marker[j] = val == 1.00;
                    }
                }
            }
            // Estimate 
            let input_reference_1: Matrix = Matrix::new(stream.clone_htod(&input_reference_1_host)?, p, n)?;
            let input_reference_2: Matrix = Matrix::new(stream.clone_htod(&input_reference_2_host)?, p, n)?;
            let shap: Matrix = network.deep_shap(input_reference_1, input_reference_2)?;
            shaps.push(shap);
            if verbose {
                pb.next();
            }
        }
        if verbose {
            pb.finish();
        }
        // Take the mean of the SHAP values across replications
        let mut shap: Matrix = shaps[0].clone();
        for i in 1..r {
            // let shap_feature_means: Vec<f32> = shaps[i].rowsummat()?.to_host()?;
            // println!("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
            // println!("i={}\nshaps[i]: {}\nshap_feature_means:{:?}", i, &shaps[i], shap_feature_means);
            shap = shap.elementwisematadd(&shaps[i])?;
        }
        shap = shap.scalarmatmul(1.00 / (r as f32))?;
        // Average the SHAP values per feature across samples or observations
        let shap_feature_means: Vec<f32> = shap.rowsummat()?.scalarmatmul(1.00/(shap.n_cols as f32))?.to_host()?;
        // Update marginal effects
        for i in 0..shap.n_rows {
            self.effects[i] = shap_feature_means[i];
        }
        // Unstandardise the effects to be more straightforward to interpret
        self.unstandaridise(network)?;
        if verbose {
            let fname_marginals_effects_png: String = self.plot(true)?;
            println!("===============================================");
            println!("Find the marginal effects (DeepSHAP estimates) barplot saved as: {}", fname_marginals_effects_png);
            println!("===============================================");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Data;
    use crate::optimisers::OptimisationParameters;
    use approx::assert_relative_eq;
    #[test]
    fn test_marginal() -> Result<(), Box<dyn Error>> {
 
        let feature_names: Vec<String> = vec!["feature_0".to_owned(), "feature_1".to_owned(),"feature_A".to_owned(),"feature_B".to_owned()];
        let marginals_order_1 = Marginals::new(feature_names.clone(), 1)?;
        println!("marginals_order_1: {:?}", marginals_order_1);
        let marginals_order_2 = Marginals::new(feature_names.clone(), 2)?;
        println!("marginals_order_2: {:?}", marginals_order_2);
        let marginals_order_3 = Marginals::new(feature_names.clone(), 3)?;
        println!("marginals_order_3: {:?}", marginals_order_3);
        // Check dimensions
        let mut marginals_tmp = Marginals::new(feature_names.clone(), 1)?;
        match marginals_tmp.check_dimensions() {
            Ok(_) => assert!(true),
            Err(_) => assert!(false),
        };
        marginals_tmp.ids.pop();
        match marginals_tmp.check_dimensions() {
            Ok(_) => assert!(false),
            Err(_) => assert!(true),
        };
        // Simulate the data and network
        let n: usize = 50; // number of observations
        let p: usize = 2; // number of continuous input features
        let q: Vec<usize> = vec![2,3]; // number of levels for each categorical feature variable
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let data = Data::simulate(n, p, q, k, n_hidden_layers, "normal", 0.0, 1.0, 42, true)?;
        let mut network = data.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        network.train(&mut optimisation_parameters, true)?;
        // Unstandardisation
        let y: Vec<f32> = data.targets.to_host()?;
        let n: f32 = y.len() as f32;
        let mean: f32 = y.iter().fold(0.0, |sum, x| sum + x) / n;
        let sd: f32 = (y.iter().fold(0.0, |sum, x| sum + (x - mean).powf(2.0)) / n).sqrt();
        let mut z: Vec<f32> = Vec::with_capacity(n as usize);
        for i in 0..(n as usize) {
            z.push((y[i] - mean) / sd);
        }
        let mut marginals_dummy = Marginals::new((0..(n as usize)).map(|x| x.to_string()).collect(), 1)?;
        marginals_dummy.effects = z;
        marginals_dummy.unstandaridise(&network)?;
        marginals_dummy.effects.iter().zip(y.iter()).for_each(|(a, b)| {assert_relative_eq!(a, b, epsilon=1.0e-6)});
        
        // Order: 1
        let mut marginals = Marginals::new(data.feature_names.clone(), 1)?;
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        marginals.estimate_perturb(&network, number_of_values_for_interpolate_between_min_and_max, true)?;
        println!("Order 1 marginals: {:?}", marginals);
        assert_eq!(marginals.ids, vec!["fcon_0", "fcon_1", "fcat_0➵0", "fcat_0➵1", "fcat_1➵0", "fcat_1➵1", "fcat_1➵2"]);
        marginals.effects.iter().zip(vec![0.009141404, 0.009156859, 0.00908426, 0.009254695, 0.009257625, 0.009228475, 0.009141026].iter()).for_each(|(a, b)| {assert_relative_eq!(a, b, epsilon=1.0e-6)});
        
        // Order: 2
        let mut marginals = Marginals::new(data.feature_names.clone(), 2)?;
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        marginals.estimate_perturb(&network, number_of_values_for_interpolate_between_min_and_max, true)?;
        println!("Order 2 marginals: {:?}", marginals);
        assert_eq!(marginals.ids.len(), 24);

        // Order: 3
        let mut marginals = Marginals::new(data.feature_names.clone(), 3)?;
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        marginals.estimate_perturb(&mut network, number_of_values_for_interpolate_between_min_and_max, true)?;
        println!("Order 3 marginals: {:?}", marginals);
        assert_eq!(marginals.ids.len(), 41);

        // DeepSHAP
        let seed: usize = 123;
        let mut marginals = Marginals::new(data.feature_names.clone(), 1)?;
        marginals.estimate_deepshap(&mut network, 100, seed, true)?;
        println!("SHAP marginals: {:?}", marginals);

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