use crate::activations::{Activation, ActivationError};
use crate::costs::{Cost, CostError};
use crate::linalg::matrix::{Matrix, MatrixError};
use crate::network::Network;
use cudarc::driver::{CudaContext, CudaSlice};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;
// use std::path::PathBuf;
// use std::env::current_dir;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, Write};
use std::io::{BufReader, BufWriter};

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
        k: usize,
        d: usize,
        seed: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let mut data = Data::new(n, p, k)?;
        let stream = data.features.data.context().default_stream();
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        let normal = Normal::new(0.0, 1.0)?;
        let features = {
            let features_host: Vec<f32> = (&mut rng).sample_iter(normal).take(p * n).collect();
            let features_dev: CudaSlice<f32> = stream.clone_htod(&features_host)?;
            Matrix::new(features_dev, p, n)?
        };
        let targets = {
            let targets_host: Vec<f32> = (&mut rng).sample_iter(normal).take(k * n).collect();
            let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_host)?;
            Matrix::new(targets_dev, k, n)?
        };
        let n_hidden_layers: usize = d;
        let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        let dropout_rates: Vec<f32> = vec![0.0; n_hidden_layers];
        let mut network = Network::new(
            &stream,
            features,
            targets,
            n_hidden_layers,
            n_hidden_nodes,
            dropout_rates,
            seed,
        )?;
        network.forwardpass()?;
        data.features = network.activations_per_layer[0].clone();
        data.targets = network.predictions.clone();
        Ok(data)
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
        // Write header
        let mut header: Vec<String> = Vec::with_capacity(p + k);
        for t in &self.target_names {
            header.push(t.clone());
        }
        for f in &self.feature_names {
            header.push(f.clone());
        }
        writeln!(writer, "{}", header.join(delim))?;
        // Write data
        for i in 0..n {
            let mut row: Vec<String> = Vec::with_capacity(p + k);
            for j in 0..k {
                row.push(format!("{}", targets[(j * n) + i]));
            }
            for j in 0..p {
                row.push(format!("{}", features[(j * n) + i]));
            }
            writeln!(writer, "{}", row.join(delim))?;
        }
        Ok(())
    }

    pub fn read_delimited(
        path: &str,
        delim: &str,
        column_indices_of_targets: Vec<usize>,
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
        if column_indices_of_targets.is_empty() {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "No column indices of targets provided.".to_string(),
            )));
        }
        if column_indices_of_targets
            .iter()
            .any(|&idx| idx >= header.len())
        {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "One or more column indices of targets are out of bounds.".to_string(),
            )));
        }
        let column_indices_features: Vec<usize> = (0..header.len())
            .filter(|idx| !column_indices_of_targets.contains(idx))
            .collect();
        let target_names: Vec<String> = column_indices_of_targets
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        let feature_names: Vec<String> = column_indices_features
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        let mut features_data: Vec<f32> = Vec::new();
        let mut targets_data: Vec<f32> = Vec::new();
        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.trim().split(delim).collect();
            if values.len() != header.len() {
                return Err(Box::new(MatrixError::DimensionMismatch(
                    "Number of values in a row does not match number of columns in header."
                        .to_string(),
                )));
            }
            for &idx in &column_indices_of_targets {
                let value: f32 = values[idx].parse()?;
                targets_data.push(value);
            }
            for &idx in &column_indices_features {
                let value: f32 = values[idx].parse()?;
                features_data.push(value);
            }
        }
        let n = targets_data.len() / column_indices_of_targets.len();
        let p = feature_names.len();
        let k = target_names.len();
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
        seed: usize,
    ) -> Result<Network, Box<dyn Error>> {
        self.check_dimensions()?;
        let stream = self.features.data.context().default_stream();
        let network = Network::new(
            &stream,
            self.features.clone(),
            self.targets.clone(),
            n_hidden_layers,
            n_hidden_nodes,
            dropout_rates,
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
    targets: Vec<f32>, // observed values (k x n)
    predictions: Vec<f32>, // predictions (k x n)
    weights_per_layer: Vec<Vec<f32>>, // weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    biases_per_layer: Vec<Vec<f32>>,  // biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    weights_x_biases_per_layer: Vec<Vec<f32>>, // summed weights (i.e. prior to activation function) ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    activations_per_layer: Vec<Vec<f32>>, // activation function output including the input layer as the first element ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    weights_gradients_per_layer: Vec<Vec<f32>>, // gradients of the weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    biases_gradients_per_layer: Vec<Vec<f32>>, // gradients of the biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    activation: String,                        // activation function enum (includes derivative)
    cost: String,                              // cost function
    seed: usize,                               // random seed for dropouts
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
        let output_data = Matrix::new(stream.clone_htod(&serdifiable_network.targets)?, k, n)?;
        let predictions = Matrix::new(stream.clone_htod(&serdifiable_network.predictions)?, k, n)?;

        let mut network: Network = Network::new(
            &stream,
            input_data,
            output_data,
            serdifiable_network.n_hidden_layers.clone(),
            serdifiable_network.n_hidden_nodes.clone(),
            serdifiable_network.dropout_rates.clone(),
            serdifiable_network.seed.clone(),
        )?;
        network.predictions = predictions;
        network.activation = match serdifiable_network.activation.as_ref() {
            "ReLU" => Activation::ReLU,
            "Sigmoid" => Activation::Sigmoid,
            "HyperbolicTangent" => Activation::HyperbolicTangent,
            _ => return Err(Box::new(ActivationError::UnimplementedActivation)),
        };
        network.cost = match serdifiable_network.cost.as_ref() {
            "MSE" => Cost::MSE,
            "MAE" => Cost::MAE,
            "HL" => Cost::HL,
            _ => return Err(Box::new(CostError::UnimplementedCost)),
        };
        for i in 0..(network.weights_per_layer.len() - 1) {
            let n_rows = serdifiable_network.n_hidden_nodes[i];
            let n_cols = if i == 0 {
                p
            } else {
                serdifiable_network.n_hidden_nodes[i - 1]
            };
            let (acti_n_rows, acti_n_cols) = if i == 0 {
                (p, n)
            } else {
                (serdifiable_network.n_hidden_nodes[i], n)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{exists, remove_file};
    #[test]
    fn test_io() -> Result<(), Box<dyn Error>> {
        let data = Data::new(100, 10, 1)?;
        let data_simulated = Data::simulate(100, 10, 1, 2, 42)?;
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

        let data_reloaded = Data::read_delimited("test_data.csv", ",", vec![0])?;
        let data_simulated_reloaded =
            Data::read_delimited("test_data_simulated.tsv", "\t", vec![0])?;

        assert!(data.features.summat()? - data_reloaded.features.summat()? < 1e-5);
        assert!(
            data_simulated.features.summat()? - data_simulated_reloaded.features.summat()? < 1e-5
        );

        println!("data_reloaded: {}", data_reloaded);
        println!("data_simulated_reloaded: {}", data_simulated_reloaded);

        // Initialise the network from reloaded data
        let network = data_simulated_reloaded.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
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

        // Clean-up
        remove_file("test_data.csv")?;
        remove_file("test_data_simulated.tsv")?;
        remove_file("test_network.json")?;

        Ok(())
    }
}
