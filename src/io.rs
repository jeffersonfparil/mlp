use crate::activations::{Activation, ActivationError};
use crate::costs::{Cost, CostError};
use crate::linalg::matrix::{Matrix, MatrixError};
use crate::network::{Network, WeightsInitialisation, NetworkError};
use crate::marginal::{Marginals, MarginalError};
use cudarc::driver::{CudaContext, CudaSlice};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Beta, Cauchy, Gamma, LogNormal, Normal, Weibull};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, Write};
use std::io::{BufReader, BufWriter};
use std::time::Instant;
use chrono::Utc;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Data {
    pub features: Matrix, // p x n: p features, n samples
    pub targets: Matrix,  // k x n: k targets, n samples
    pub feature_names: Vec<String>,
    pub target_names: Vec<String>,
}

/// Implement std::fmt::Display for MatrixError
impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Data {{\n  features: {},\n  targets: {}\n}}",
            self.features, self.targets
        )
    }
}

fn simulate_weights(dist: &str, par1: f64, par2: f64, p: usize, seed: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
    let weights: Vec<f32> = match dist {
        "normal" => {
            let distribution = Normal::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        "lognormal" => {
            let distribution = LogNormal::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        "cauchy" => {
            let distribution = Cauchy::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        "weibull" => {
            let distribution = Weibull::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        "gamma" => {
            let distribution = Gamma::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        "beta" => {
            let distribution = Beta::new(par1, par2)?;
            (&mut rng)
                .sample_iter(distribution)
                .take(p)
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        },
        _ => {
            let distribution = Normal::new(par1, par2)?;
            // (&mut rng)
            //     .sample_iter(distribution)
            //     .take(p)
            //     .map(|x| x as f32)
            //     .collect::<Vec<f32>>()
            let mut b: Vec<f32> = Vec::with_capacity(p);
            let step_size: usize = 1_000_000;
            for j in (0..p).step_by(step_size) {
                let m = if j+step_size > p {
                    p - j
                } else {
                    step_size
                };
                let tmp: Vec<f32> = (&mut rng).sample_iter(distribution).take(m) .map(|x| x as f32).collect();
                b.extend(&tmp);
            }
            b
        },
    };
    Ok(weights)
}

impl Data {
    pub fn new(n: usize, p: usize, k: usize) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let features_dev: CudaSlice<f32> = stream.clone_htod(&vec![0.0f32; p * n])?;
        let targets_dev: CudaSlice<f32> = stream.clone_htod(&vec![0.0f32; k * n])?;
        let features = Matrix::new(features_dev, p, n)?;
        let targets = Matrix::new(targets_dev, k, n)?;
        let feature_names: Vec<String> = (0..p).map(|i| format!("feature_{}", i)).collect();
        let target_names: Vec<String> = (0..k).map(|i| format!("target_{}", i)).collect();
        Ok(Data {
            features,
            targets,
            feature_names,
            target_names,
        })
    }

    pub fn simulate(
        n: usize,
        p: usize,
        q: Vec<usize>,
        k: usize,
        d: usize,
        dist: &str,
        par1: f64,
        par2: f64,
        seed: usize,
        verbose: bool,
    ) -> Result<(Self, Network), Box<dyn Error>> {
        // n = total number of observations
        // p = number of continuous explanatory variables or features
        // q = vector of the number of levels in categorical variable
        // k = number of response variables or targets
        // d = number of hidden layers
        // dist = distribution of the weights (all biases will be set to zero for simplicity + all distributions will have 2 controllable parameters)
        // par1 = first parameter of the weights distributions, e.g. mean for Normal distribution, and shape for Gamma distribution
        // par2 = second parameter of the weights distributions, e.g. standard deviation for Normal distribution, and scale for Gamma distribution
        // seed = randomisation seed for repeatability

        if verbose {println!("(1/8) Simulating feature ids...")}
        let time = Instant::now();
        let n_features_categorical = q.iter().fold(0, |sum, &x| sum + x);
        let n_features = p + n_features_categorical;
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        // Features simulation
        let mut feature_names: Vec<String> = Vec::with_capacity(n_features);
        let mut features_host: Vec<f32> = Vec::with_capacity(n_features * n);
        // Continuous features
        for j in 0..p {
            feature_names.push(format!("fcon_{}", j));
            for _i in 0..n {
                features_host.push(rng.random());
            }
        }
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Categorical features (one-hot encoded) exploring all level combinations
        if verbose {println!("(2/8) Simulating categorical features (if any) ...")}
        let time = Instant::now();
        if n_features_categorical > 0 {
            let n_combinations = q.iter().product::<usize>(); // Calculate total number of combinations by multiplying all levels in q
            let mut categorical_levels: Vec<Vec<usize>> = vec![vec![0; n]; q.len()]; // Initialize a vector of vectors to store levels for each categorical variable, each with n observations
            for i in 0..n { // For each observation
                let mut combo_index = i % n_combinations; // Get the combination index for this observation, cycling through all combinations if n > n_combinations
                for (id, &n_levels) in q.iter().enumerate() { // For each categorical variable
                    categorical_levels[id][i] = combo_index % n_levels; // Assign the level for this variable in this observation by taking modulo of the current combo_index
                    // println!("i={}' combo_index={} id={} n_levels={} level={}", i, combo_index, id, n_levels, categorical_levels[id][i]);
                    combo_index /= n_levels; // Divide combo_index by n_levels to shift to the next variable's level (like mixed radix conversion)
                }
            }
            for (id, &n_levels) in q.iter().enumerate() {
                for j in 0..n_levels {
                    feature_names.push(format!("fcat_{}➵{}", id, j));
                    for i in 0..n {
                        features_host.push(if categorical_levels[id][i] == j {
                            1.0
                        } else {
                            0.0
                        });
                    }
                }
            }
        }
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Dummy targets, i.e. prior to simulating the weights as the initiator for Network uses He initialisation (sampling from a normal distribution)
        if verbose {println!("(3/8) Simulating dummy targets...")}
        let time = Instant::now();
        let targets_host: Vec<f32> = (0..(k*n)).map(|_| rng.random()).collect();
        // println!("n = {}", n);
        // println!("p = {}", p);
        // println!("k = {}", k);
        // println!("n_features = {}", n_features);
        // println!("q = {:?}", q);
        // println!("features_host.len() = {}", features_host.len());
        // println!("targets_host.len() = {}", targets_host.len());
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Instantiate the Data and extract the CUDA device stream for instantiating the features and target matrices
        if verbose {println!("(4/8) Simulating Data struct...")}
        let time = Instant::now();
        let mut data = Data::new(n, n_features, k)?;
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Instantiate the Network
        let stream = data.features.data.context().default_stream();
        let features: Matrix = Matrix::new(stream.clone_htod(&features_host)?, n_features, n)?;
        // println!("features={}", features);
        let targets: Matrix = Matrix::new(stream.clone_htod(&targets_host)?, k, n)?;
        // println!("targets={}", targets);
        let n_hidden_layers: usize = d;
        let n_hidden_nodes: Vec<usize> = vec![(n_features as f64 / 2.0).ceil() as usize; n_hidden_layers]; // we use half the number of input features as the number of nodes in the hidden layers
        let dropout_rates: Vec<f32> = vec![0.0; n_hidden_layers];
        if verbose {println!("(5/8) Simulating Network struct...")}
        let time = Instant::now();
        let mut network = Network::new(
            &stream,
            features,
            targets,
            n_hidden_layers,
            n_hidden_nodes,
            dropout_rates,
            WeightsInitialisation::He,
            seed,
        )?;
        // println!("network: {}", network);
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Redefine the weights
        if verbose {println!("(6/8) Simulating weights and replacing the ones initialised in the Network struct...")}
        let time = Instant::now();
        let dummy_dev: Matrix = Matrix::new(stream.clone_htod(&vec![0.0])?, 1, 1)?;
        for i in 0..(network.n_hidden_layers+1) {
            let n_rows = network.weights_per_layer[i].n_rows;
            let n_cols = network.weights_per_layer[i].n_cols;
            let m = n_rows * n_cols;
            let weights_host: Vec<f32> = simulate_weights(dist, par1, par2, m, seed)?;
            // let weights: Matrix = Matrix::new(stream.clone_htod(&weights_host)?, network.weights_per_layer[i].n_rows, network.weights_per_layer[i].n_cols)?;
            // println!("i={}; weights={}", i, weights);
            network.weights_per_layer[i] = dummy_dev.clone(); // to release some GPU memory before replacing the weights
            network.weights_per_layer[i] = Matrix::new(stream.clone_htod(&weights_host)?, n_rows, n_cols)?;
        }
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Extract non-dummy targets
        if verbose {println!("(7/8) Simulating non-dummy targets...")}
        let time = Instant::now();
        network.predict()?;
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        // Update the feature names
        if verbose {println!("(8/8) Outputing the final simulated Data struct...")}
        let time = Instant::now();
        data.feature_names = feature_names.clone();
        // Update the features and targets with the simulated data
        data.features = network.activations_per_layer[0].clone();
        data.targets = network.predictions.clone();
        if verbose {println!("\t→ {:.2} minutes\n", time.elapsed().as_millis() as f64 / 60_000.0)};
        Ok((data, network))
    }

    pub fn check_dimensions(&self) -> Result<(), MatrixError> {
        if self.features.n_cols != self.targets.n_cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of observations (n_cols) in features ({}) does not match number of observations (n_cols) in targets ({}).",
                self.features.n_cols, self.targets.n_cols
            )));
        }
        if self.features.n_rows == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of features (n_rows) is zero."
            )));
        }
        if self.targets.n_rows == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of target variable/s (n_rows) is zero."
            )));
        }
        if self.feature_names.len() != self.features.n_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of feature names ({}) does not match number of features ({}).",
                self.feature_names.len(),
                self.features.n_rows
            )));
        }
        if self.target_names.len() != self.targets.n_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of target names ({}) does not match number of target variable/s ({}).",
                self.target_names.len(),
                self.targets.n_rows
            )));
        }
        Ok(())
    }

    pub fn write_delimited(&self, path: &str, delim: &str) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let features = self.features.to_host()?;
        let targets = self.targets.to_host()?;
        let file = File::create_new(path)?; // makes sure not to overwrite existing files, i.e. using create_new() instead of just create()
        let mut writer = BufWriter::new(file);
        let n = self.features.n_cols;
        let p = self.features.n_rows;
        let k = self.targets.n_rows;
        // Define the number of continuous features plus the number of levels in each categorical feature
        // Also extract the header names, i.e.:
        //      - names of the target variables,
        //      - names of the continuous features, and
        //      - base names of the categorical features
        let mut n_features: usize = 0;
        let mut header: Vec<String> = Vec::new();
        for t in &self.target_names {
            header.push(t.clone());
        }
        for f in &self.feature_names {
            let f_split = f.split("➵").collect::<Vec<&str>>();
            if f_split.len() == 1 {
                // Continuous features
                header.push(f.clone());
            } else {
                // Categorical features
                // Assumes the categorical factor levels are sorted/grouped together along the vector of feature names
                if header[header.len()-1] != f_split[0] {
                    // Add the categorical feature if it is new
                    header.push(f_split[0].to_owned());
                    n_features += 1;
                }
            }
        }
        // Write header
        writeln!(writer, "{}", header.join(delim))?;
        // Write data
        for i in 0..n {
            let mut row: Vec<String> = Vec::with_capacity(k + n_features);
            // Write targets
            for j in 0..k {
                row.push(format!("{}", targets[(j * n) + i]));
            }
            // Write features
            for j in 0..p {
                let val: f32 = features[(j * n) + i];
                let f_split: Vec<&str> = self.feature_names[j].split("➵").collect::<Vec<&str>>();
                if f_split.len() == 1 {
                    row.push(format!("{}", val));
                } else {
                    if val == 1.00 {
                        // Add "level" if the categorical feature level is numeric
                        match f_split[1].parse::<f64>() {
                            Ok(_) => {row.push(format!("level-{}", f_split[1]));},
                            Err(_) => {row.push(format!("{}", f_split[1]));}
                        };
                    }
                }
            }
            writeln!(writer, "{}", row.join(delim))?;
        }
        Ok(())
    }

    pub fn read_delimited(
        path: &str,
        delim: &str,
        column_indices_targets: &Vec<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        // Read header
        let header: Vec<String> = if let Some(header_line) = lines.next() {
            let header = header_line?;
            header.trim().split(delim).map(|s| s.to_string()).collect()
        } else {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "File is empty.".to_string(),
            )));
        };
        if column_indices_targets.is_empty() {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "No column indices of targets provided.".to_string(),
            )));
        }
        if column_indices_targets
            .iter()
            .any(|&idx| idx >= header.len())
        {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "One or more column indices of targets are out of bounds.".to_string(),
            )));
        }
        let column_indices_features: Vec<usize> = (0..header.len())
            .filter(|idx| !column_indices_targets.contains(idx))
            .collect();
        let target_names_tmp: Vec<String> = column_indices_targets
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        let feature_names_tmp: Vec<String> = column_indices_features
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        // Convert targets and features into one-hot encoding if they are non-numeric
        // First we do a first pass to identify all levels per variable
        // And on the second pass we extract the one-hot encoding matrices
        enum Value {
            Numeric(f32),
            Text(String),
        }
        let mut features_data_tmp: Vec<Value> = Vec::new();
        let mut targets_data_tmp: Vec<Value> = Vec::new();
        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.trim().split(delim).collect();
            if values.len() != header.len() {
                return Err(Box::new(MatrixError::DimensionMismatch(
                    "Number of values in a row does not match number of columns in header."
                        .to_string(),
                )));
            }
            for &idx in column_indices_targets {
                let value = match values[idx].parse::<f32>() {
                    Ok(x) => Value::Numeric(x),
                    Err(_) => Value::Text(values[idx].to_owned()),
                };
                targets_data_tmp.push(value);
            }
            for &idx in &column_indices_features {
                let value = match values[idx].parse::<f32>() {
                    Ok(x) => Value::Numeric(x),
                    Err(_) => Value::Text(values[idx].to_owned()),
                };
                features_data_tmp.push(value);
            }
        }
        // Count the number of levels of the non-numeric variables in preparation for one-hot encoding
        let n = targets_data_tmp.len() / column_indices_targets.len();
        let p = feature_names_tmp.len();
        let k = target_names_tmp.len();
        // println!("target_names_tmp: {:?}", target_names_tmp);
        // println!("feature_names_tmp: {:?}", feature_names_tmp);
        // Targets levels
        let mut targets_levels: Vec<Vec<String>> = Vec::with_capacity(k);
        for j in 0..k {
            targets_levels.push(Vec::new());
            for i in 0..n {
                let idx = i*k + j;
                match &targets_data_tmp[idx] {
                    Value::Numeric(x) => {
                        if (targets_levels[j].len() > 0) & (!targets_levels[j].contains(&x.to_string())) {
                            targets_levels[j].push(x.to_string());
                        } else {
                            // Note: we assume that the first 100 elements of the non-numeric target variable cannot be parsed as numeric
                            if idx < 100 {
                                continue
                            } else {
                                break
                            }
                        }
                    },
                    Value::Text(x) => {
                        if !targets_levels[j].contains(x) {
                            targets_levels[j].push(x.to_owned());
                        }
                    }
                };
            }
        }
        // println!("targets_levels: {:?}", targets_levels);
        // Features levels
        let mut features_levels: Vec<Vec<String>> = Vec::with_capacity(p);
        for j in 0..p {
            features_levels.push(Vec::new());
            for i in 0..n {
                let idx = i*p + j;
                match &features_data_tmp[idx] {
                    Value::Numeric(x) => {
                        if (features_levels[j].len() > 0) & (!features_levels[j].contains(&x.to_string())) {
                            features_levels[j].push(x.to_string());
                        } else {
                            // Note: we assume that the first element of the non-numeric target variable cannot be parsed as numeric
                            break;
                        }
                    },
                    Value::Text(x) => {
                        if !features_levels[j].contains(x) {
                            features_levels[j].push(x.to_owned());
                        }
                    }
                };
            }
        }
        // println!("features_levels: {:?}", features_levels);
        // Build the one-hot encodings of the targets and/or features
        // Targets values
        let m = targets_levels
            .iter()
            .fold(0, |sum, x| {
                if x.len() == 0 {
                    sum + 1
                } else {
                    sum + x.len()
                }
            });
        let mut targets_data: Vec<f32> = vec![0.0; m*n];
         for j in 0..k {
            let m_tmp = targets_levels[0..(j+1)]
            .iter()
            .fold(0, |sum, x| {
                if x.len() == 0 {
                    sum + 1
                } else {
                    sum + x.len()
                }
            });
            // println!("m={}", m);
            // println!("m_tmp={}", m_tmp);
            for i in 0..n {
                let idx_source = i*k + j;
                if targets_levels[j].len() == 0 {
                    // Numerics
                    let idx_destination = (m_tmp-1)*n + i;
                    targets_data[idx_destination] = match &targets_data_tmp[idx_source] {
                        Value::Numeric(x) => *x,
                        Value::Text(_) => {
                            return Err(Box::new(MatrixError::TypeMismatch(
                                format!("Unexpected type mismatch in target variable: {}. We expected a numeric variable. Please remove rows with missing data.", target_names_tmp[j])
                            )));
                        },
                    };
                } else {
                    // Non-numerics
                    match &targets_data_tmp[idx_source] {
                        Value::Text(x) => {
                            let mut idx: usize = 0;
                            for i_tmp in 0..targets_levels[j].len() {
                                if targets_levels[j][i_tmp] == x.to_owned() {
                                    idx = i_tmp;
                                    break
                                }
                            }
                            let idx_destination = (m_tmp-(targets_levels[j].len() - idx))*n + i;
                            targets_data[idx_destination] = 1.0;
                        },
                        Value::Numeric(_) => {
                            return Err(Box::new(MatrixError::TypeMismatch(
                                format!("Unexpected type mismatch in target variable: {}. We expected a non-numeric variable. Please remove rows with missing data.", target_names_tmp[j])
                            )));
                        },
                    };
                }
            }
        }
        // println!("targets_data: {:?}", targets_data);
        // Features values
        let m = features_levels
            .iter()
            .fold(0, |sum, x| {
                if x.len() == 0 {
                    sum + 1
                } else {
                    sum + x.len()
                }
            });
        let mut features_data: Vec<f32> = vec![0.0; m*n];
         for j in 0..p {
            let m_tmp = features_levels[0..(j+1)]
            .iter()
            .fold(0, |sum, x| {
                if x.len() == 0 {
                    sum + 1
                } else {
                    sum + x.len()
                }
            });
            // println!("m={}", m);
            // println!("m_tmp={}", m_tmp);
            for i in 0..n {
                let idx_source = i*p + j;
                if features_levels[j].len() == 0 {
                    // Numerics
                    let idx_destination = (m_tmp-1)*n + i;
                    features_data[idx_destination] = match &features_data_tmp[idx_source] {
                        Value::Numeric(x) => *x,
                        Value::Text(_) => {
                            return Err(Box::new(MatrixError::TypeMismatch(
                                format!("Unexpected type mismatch in feature variable: {}. We expected a numeric variable.", feature_names_tmp[j])
                            )));
                        },
                    };
                } else {
                    // Non-numerics
                    match &features_data_tmp[idx_source] {
                        Value::Text(x) => {
                            let mut idx: usize = 0;
                            for i_tmp in 0..features_levels[j].len() {
                                if features_levels[j][i_tmp] == x.to_owned() {
                                    idx = i_tmp;
                                    break
                                }
                            }
                            let idx_destination = (m_tmp-(features_levels[j].len() - idx))*n + i;
                            features_data[idx_destination] = 1.0;
                        },
                        Value::Numeric(_) => {
                            return Err(Box::new(MatrixError::TypeMismatch(
                                format!("Unexpected type mismatch in feature variable: {}. We expected a non-numeric.", feature_names_tmp[j])
                            )));
                        },
                    };
                }
            }
        }
        // println!("features_data: {:?}", features_data);
        // Update target_names and feature_names
        let k = targets_data.len() / n;
        let p = features_data.len() / n;
        let mut target_names: Vec<String> = Vec::with_capacity(k);
        let mut feature_names: Vec<String> = Vec::with_capacity(p);
        for i in 0..target_names_tmp.len() {
            let name: String = target_names_tmp[i].to_owned();
            if targets_levels[i].len() == 0 {
                target_names.push(name);
            } else {
                for j in 0..targets_levels[i].len() {
                    let new_name = format!("{}➵{}", name, targets_levels[i][j].to_owned());
                    target_names.push(new_name);
                }
            }
        }
        for i in 0..feature_names_tmp.len() {
            let name: String = feature_names_tmp[i].to_owned();
            if features_levels[i].len() == 0 {
                feature_names.push(name);
            } else {
                for j in 0..features_levels[i].len() {
                    let new_name = format!("{}➵{}", name, features_levels[i][j].to_owned());
                    feature_names.push(new_name);
                }
            }
        }
        // println!("target_names_tmp: {:?}", target_names_tmp);
        // println!("feature_names_tmp: {:?}", feature_names_tmp);
        // println!("target_names: {:?}", target_names);
        // println!("feature_names: {:?}", feature_names);
        let mut data = Data::new(n, p, k)?;
        let stream = data.features.data.context().default_stream();
        let features_dev: CudaSlice<f32> = stream.clone_htod(&features_data)?;
        let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_data)?;
        data.features = Matrix::new(features_dev, p, n)?;
        data.targets = Matrix::new(targets_dev, k, n)?;
        data.feature_names = feature_names;
        data.target_names = target_names;
        Ok(data)
    }
    
    pub fn init_network(
        &self,
        n_hidden_layers: usize,
        n_hidden_nodes: Vec<usize>,
        dropout_rates: Vec<f32>,
        weights_initialisation: WeightsInitialisation,
        seed: usize,
    ) -> Result<Network, Box<dyn Error>> {
        self.check_dimensions()?;
        let stream = self.features.data.context().default_stream();
        let network = Network::new(
            &stream,
            self.features.clone(),
            self.targets.clone(),
            n_hidden_layers,
            n_hidden_nodes.clone(),
            dropout_rates,
            weights_initialisation,
            seed,
        )?;
        Ok(network)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SerdifiableNetwork {
    n_observations: usize, // number of observations, i.e number of columns in targets, predictions, first element in weights_x_biases per layer (pre-activation layers) and first element in activations_per_layer
    n_features: usize, // number of input features, i.e. number of columns in the first layers of weights and its gradients
    n_targets: usize, // number of dimensions of the output data, i.e. number of rows in targets, predictions, last element in weights_x_biases per layer (pre-activation layers) and last element in activations per layer
    n_hidden_layers: usize, // number of hidden layers
    n_hidden_nodes: Vec<usize>, // number of nodes per hidden layer (k)
    dropout_rates: Vec<f32>, // soft dropout rates per hidden layer (k)
    targets: Vec<f32>, // observed values (k x n; standardised)
    targets_mean_sd: (f32,f32), // mean and standard deviation of the target values across k rows
    predictions: Vec<f32>, // predictions (k x n)
    weights_per_layer: Vec<Vec<f32>>, // weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    biases_per_layer: Vec<Vec<f32>>,  // biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    weights_x_biases_per_layer: Vec<Vec<f32>>, // summed weights (i.e. prior to activation function) ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    activations_per_layer: Vec<Vec<f32>>, // activation function output including the input layer as the first element ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    weights_gradients_per_layer: Vec<Vec<f32>>, // gradients of the weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    biases_gradients_per_layer: Vec<Vec<f32>>, // gradients of the biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    activation: String, // activation function enum (includes derivative)
    cost: String, // cost function
    weights_initialisation: String, // weights initialisation, i.e. He, Cauchy, Uniform or StandardNormal
    n_epochs: usize, // number of training epochs
    seed: usize, // random seed for dropouts
    loss: f32, // mean loss (additional field not part of the actual Network struct)
}

impl Network {
    pub fn save_network(&self, fname: &str) -> Result<(), Box<dyn Error>> {
        let serdifiable_network = SerdifiableNetwork {
            n_observations: self.targets.n_cols,
            n_features: self.weights_per_layer[0].n_cols,
            n_targets: self.targets.n_rows,
            n_hidden_layers: self.n_hidden_layers.clone(),
            n_hidden_nodes: self.n_hidden_nodes.clone(),
            dropout_rates: self.dropout_rates.clone(),
            targets: self.targets.to_host()?,
            targets_mean_sd: self.targets_mean_sd,
            predictions: self.predictions.to_host()?,
            weights_per_layer: self
                .weights_per_layer
                .iter()
                .map(|x| x.to_host().expect("Error extracting weights per layer"))
                .collect(),
            biases_per_layer: self
                .biases_per_layer
                .iter()
                .map(|x| x.to_host().expect("Error extracting biases per layer"))
                .collect(),
            weights_x_biases_per_layer: self
                .weights_x_biases_per_layer
                .iter()
                .map(|x| {
                    x.to_host()
                        .expect("Error extracting pre-activations per layer")
                })
                .collect(),
            activations_per_layer: self
                .activations_per_layer
                .iter()
                .map(|x| x.to_host().expect("Error extracting activations per layer"))
                .collect(),
            weights_gradients_per_layer: self
                .weights_gradients_per_layer
                .iter()
                .map(|x| {
                    x.to_host()
                        .expect("Error extracting weights gradients per layer")
                })
                .collect(),
            biases_gradients_per_layer: self
                .biases_gradients_per_layer
                .iter()
                .map(|x| {
                    x.to_host()
                        .expect("Error extracting biases gradients per layer")
                })
                .collect(),
            activation: self.activation.to_string(),
            cost: self.cost.to_string(),
            weights_initialisation: self.weights_initialisation.to_string(),
            n_epochs: self.n_epochs,
            seed: self.seed,
            loss: self.loss()?,
        };
        let json_data = serde_json::to_string_pretty(&serdifiable_network)?;
        let mut file = File::create_new(fname)?; // makes sure not to overwrite existing files, i.e. using create_new() instead of just create()
        file.write_all(json_data.as_bytes())?;
        Ok(())
    }

    pub fn read_network(fname: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(fname)?;
        let reader = BufReader::new(file);
        let serdifiable_network: SerdifiableNetwork = serde_json::from_reader(reader)?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let n = serdifiable_network.n_observations;
        let p = serdifiable_network.n_features;
        let k = serdifiable_network.n_targets;

        let input_data = Matrix::new(
            stream.clone_htod(&serdifiable_network.activations_per_layer[0])?,
            p,
            n,
        )?;
        let unstandardised_output_data: Vec<f32> = serdifiable_network.targets.iter().map(|&x| (x * serdifiable_network.targets_mean_sd.1) + serdifiable_network.targets_mean_sd.0).collect();
        let output_data = Matrix::new(stream.clone_htod(&unstandardised_output_data)?, k, n)?;
        let predictions = Matrix::new(stream.clone_htod(&serdifiable_network.predictions)?, k, n)?;
        let weights_initialisation = match serdifiable_network.weights_initialisation.as_ref() {
            "He" => WeightsInitialisation::He,
            "Cauchy" => WeightsInitialisation::Cauchy,
            "Uniform" => WeightsInitialisation::Uniform,
            "StandardNormal" => WeightsInitialisation::StandardNormal,
            e => return Err(Box::new(NetworkError::OtherError(format!("Unrecognised weights initialisation: {}", e)))),
        };
        let mut network: Network = Network::new(
            &stream,
            input_data,
            output_data,
            serdifiable_network.n_hidden_layers.clone(),
            serdifiable_network.n_hidden_nodes.clone(),
            serdifiable_network.dropout_rates.clone(),
            weights_initialisation,
            serdifiable_network.seed.clone(),
        )?;
        network.predictions = predictions;
        network.activation = match serdifiable_network.activation.as_ref() {
            "ReLU" => Activation::ReLU,
            "Sigmoid" => Activation::Sigmoid,
            "HyperbolicTangent" => Activation::HyperbolicTangent,
            "Linear" => Activation::Linear,
            _ => return Err(Box::new(ActivationError::UnimplementedActivation)),
        };
        network.cost = match serdifiable_network.cost.as_ref() {
            "MSE" => Cost::MSE,
            "MAE" => Cost::MAE,
            "HL" => Cost::HL,
            _ => return Err(Box::new(CostError::UnimplementedCost)),
        };
        for i in 0..network.weights_per_layer.len() {
            let n_rows = if i == (network.weights_per_layer.len() - 1) {
                k
            } else {
                serdifiable_network.n_hidden_nodes[i]
            };
            let n_cols = if i == 0 {
                p
            } else {
                serdifiable_network.n_hidden_nodes[i - 1]
            };
            let (acti_n_rows, acti_n_cols) = if i == 0 {
                (p, n)
            } else {
                (serdifiable_network.n_hidden_nodes[i-1], n)
            };
            network.weights_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.weights_per_layer[i])?,
                n_rows,
                n_cols,
            )?;
            network.biases_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.biases_per_layer[i])?,
                n_rows,
                1,
            )?;
            network.weights_x_biases_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.weights_x_biases_per_layer[i])?,
                n_rows,
                n,
            )?;
            network.activations_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.activations_per_layer[i])?,
                acti_n_rows,
                acti_n_cols,
            )?;
            network.weights_gradients_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.weights_gradients_per_layer[i])?,
                n_rows,
                n_cols,
            )?;
            network.biases_gradients_per_layer[i] = Matrix::new(
                stream.clone_htod(&serdifiable_network.biases_gradients_per_layer[i])?,
                n_rows,
                1,
            )?;
        }
        Ok(network)
    }
}

impl Marginals {
    pub fn write_delimited(&self, path: &str, delim: &str) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let file = File::create_new(path)?; // makes sure not to overwrite existing files, i.e. using create_new() instead of just create()
        let mut writer = BufWriter::new(file);
        let n = self.ids.len();
        // Write header
        writeln!(writer, "{}", vec!["ids", "effects", "r2s"].join(delim))?;
        // Write data
        for i in 0..n {
            let row: Vec<String> = vec![self.ids[i].to_owned(), self.effects[i].to_string(), self.r2s[i].to_string()];
            writeln!(writer, "{}", row.join(delim))?;
        }
        Ok(())
    }

    pub fn read_delimited(fname: &str, delim: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(fname)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        // Read header
        let header: Vec<String> = if let Some(header_line) = lines.next() {
            let header = header_line?;
            header.trim().split(delim).map(|s| s.to_string()).collect()
        } else {
            return Err(Box::new(MarginalError::DimensionMismatch(
                "File is empty.".to_string(),
            )));
        };
        if &header[0] != "ids" {
            return Err(Box::new(MarginalError::NameMismatch(
                format!("We expect the first column to be \"ids\" but found \"{}\" instead.", header[0]),
            )));
        }
        if &header[1] != "effects" {
            return Err(Box::new(MarginalError::NameMismatch(
                format!("We expect the second column to be \"effects\" but found \"{}\" instead.", header[1]),
            )));
        }
        if &header[2] != "r2s" {
            return Err(Box::new(MarginalError::NameMismatch(
                format!("We expect the third column to be \"r2s\" but found \"{}\" instead.", header[1]),
            )));
        }
        let mut ids: Vec<String> = Vec::new();
        let mut effects: Vec<f32> = Vec::new();
        let mut r2s: Vec<f32> = Vec::new();
        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.trim().split(delim).collect();
            if values.len() != header.len() {
                return Err(Box::new(MarginalError::DimensionMismatch(
                    "Number of values in a row does not match number of columns in header."
                        .to_string(),
                )));
            }
            ids.push(values[0].to_owned());
            effects.push(values[1].parse::<f32>()?);
            r2s.push(values[2].parse::<f32>()?);
        }
        let marginals =  Marginals {ids, effects, r2s};
        marginals.check_dimensions()?;
        Ok(marginals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{exists, remove_file};
    #[test]
    fn test_io() -> Result<(), Box<dyn Error>> {
        let data = Data::new(100, 10, 1)?;
        let (data_simulated, _network_simulated) = Data::simulate(100, 5, vec![2,3], 1, 2, "normal", 0.0, 1.0, 42, true)?;
        assert_eq!(data.features.n_rows, data_simulated.features.n_rows);
        assert!(data.targets.summat()? == 0.0);
        assert!(data_simulated.targets.summat()? != 0.0);
        assert!(data.features.summat()? == 0.0);
        assert!(data_simulated.features.summat()? != 0.0);
        assert_eq!(data.check_dimensions(), Ok(()));
        println!("data: {}", data);
        println!("data_simulated: {}", data_simulated);
        if exists("test_data.csv")? {
            remove_file("test_data.csv")?;
        }
        if exists("test_data_simulated.tsv")? {
            remove_file("test_data_simulated.tsv")?;
        }
        data.write_delimited("test_data.csv", ",")?;
        data_simulated.write_delimited("test_data_simulated.tsv", "\t")?;
        let data_reloaded = Data::read_delimited("test_data.csv", ",", &vec![0])?;
        let data_simulated_reloaded =
            Data::read_delimited("test_data_simulated.tsv", "\t", &vec![0])?;
        // Check full contents: test_data_simulated.tsv and test_data_simulated_rewritten.tsv they should be identical
        data_simulated_reloaded.write_delimited("test_data_simulated_rewritten.tsv", "\t")?;
        assert_eq!(
            std::fs::read_to_string("test_data_simulated.tsv")?,
            std::fs::read_to_string("test_data_simulated_rewritten.tsv")?
        );
        assert!(data.features.summat()? - data_reloaded.features.summat()? < 1e-5);
        assert!(
            data_simulated.features.summat()? - data_simulated_reloaded.features.summat()? < 1e-5
        );
        println!("data_reloaded: {}", data_reloaded);
        println!("data_simulated_reloaded: {}", data_simulated_reloaded);
        // Initialise the network from reloaded data
        let mut network = data_simulated_reloaded.init_network(2, vec![5; 2], vec![0.0; 2], WeightsInitialisation::He, 42)?;
        assert!(network.targets.summat()? - data_simulated_reloaded.targets.summat()? < 1e-5);
        assert!(
            network.activations_per_layer[0].summat()?
                - data_simulated_reloaded.features.summat()?
                < 1e-5
        );
        assert_eq!(network.n_hidden_layers, 2);
        println!("network: {}", network);
        if exists("test_network.json")? {
            remove_file("test_network.json")?;
        }
        network.save_network("test_network.json")?;
        let network_reloaded = Network::read_network("test_network.json")?;
        println!("network_reloaded={}", network_reloaded);
        assert_eq!(
            network.check_dimensions()?,
            network_reloaded.check_dimensions()?
        );
        assert_eq!(
            network.predictions.summat()?,
            network_reloaded.predictions.summat()?
        );
        // Data with non-numerics
        let fname_non_numerics: String = "test_non_numerics.csv".to_owned();
        {
            let file = File::create_new(&fname_non_numerics)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "{}", "target_0,target_1,feature_0,feature_1,feature_2,feature_3")?;
            writeln!(writer, "{}", "A,0.002356832,X,A1,0.26257637,-0.22530088")?;
            writeln!(writer, "{}", "B,0.009485791,Y,A2,-0.40898767,-0.6339346")?;
            writeln!(writer, "{}", "C,0.009100225,Z,A3,0.012834634,-2.0523884")?;
            writeln!(writer, "{}", "C,0.004334052,Z,A4,1.0629518,2.0183794")?;
            writeln!(writer, "{}", "C,0.015800802,Z,A4,-0.13212654,-1.7721263")?;
            writeln!(writer, "{}", "C,0.002177081,Z,A4,0.39454332,-0.8285658")?;
            writeln!(writer, "{}", "C,0.021280818,Z,A4,-0.15998206,0.07512082")?;
            writeln!(writer, "{}", "C,0.02473503,A,A3,1.6373256,0.27236217")?;
            writeln!(writer, "{}", "C,0.019157464,B,A2,-0.6462233,0.92315364")?;
            writeln!(writer, "{}", "D,0.016854811,C,A1,0.34480542,0.534274")?;
        }
        let data_reloaded = Data::read_delimited(&fname_non_numerics.as_str(), ",", &vec![0])?;
        data_reloaded.write_delimited("test_non_numerics_rewritten.csv", ",")?;
        let data_rewritten = Data::read_delimited("test_non_numerics_rewritten.csv", ",", &vec![0])?;
        // Check full contents: re-written non-numerics should match our expectations
        println!("data_reloaded: {}", data_reloaded);
        println!("data_rewritten: {}", data_rewritten);
        assert_eq!(
            std::fs::read_to_string("test_non_numerics_rewritten.csv")?,
            "target_0➵A,target_0➵B,target_0➵C,target_0➵D,target_1,feature_0,feature_1,feature_2,feature_3\n1,0,0,0,0.002356832,X,A1,0.26257637,-0.22530088\n0,1,0,0,0.009485791,Y,A2,-0.40898767,-0.6339346\n0,0,1,0,0.009100225,Z,A3,0.012834634,-2.0523884\n0,0,1,0,0.004334052,Z,A4,1.0629518,2.0183794\n0,0,1,0,0.015800802,Z,A4,-0.13212654,-1.7721263\n0,0,1,0,0.002177081,Z,A4,0.39454332,-0.8285658\n0,0,1,0,0.021280818,Z,A4,-0.15998206,0.07512082\n0,0,1,0,0.02473503,A,A3,1.6373256,0.27236217\n0,0,1,0,0.019157464,B,A2,-0.6462233,0.92315364\n0,0,0,1,0.016854811,C,A1,0.34480542,0.534274\n".to_owned(),
        );
        // Marginals
        let mut marginals = Marginals::new(data.feature_names.clone(), 3)?;
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        marginals.estimate_perturb(&mut network, number_of_values_for_interpolate_between_min_and_max, true)?;
        marginals.write_delimited("test_marginals.tsv", "\t")?;
        let marginals_reloaded = Marginals::read_delimited("test_marginals.tsv", "\t")?;
        // println!("marginals: {:?}", marginals);
        // println!("marginals_reloaded: {:?}", marginals_reloaded);
        assert_eq!(marginals, marginals_reloaded);
        // Clean-up
        for f in std::fs::read_dir(".")? {
            let f = f?.path();
            if f.is_file() && 
            (
                f.extension().and_then(|s| s.to_str()) == Some("png") || 
                f.extension().and_then(|s| s.to_str()) == Some("svg") || 
                f.extension().and_then(|s| s.to_str()) == Some("json") || 
                f.extension().and_then(|s| s.to_str()) == Some("csv") || 
                f.extension().and_then(|s| s.to_str()) == Some("tsv") 
            ) {
                std::fs::remove_file(&f)?;
            }
        }
        Ok(())
    }
}
