use crate::linalg::matrix::Matrix;
use crate::network::Network;
use cudarc::driver::CudaSlice;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use std::error::Error;

impl Network {
    pub fn dropout(&mut self, i: usize) -> Result<(), Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64((self.seed + i + self.n_epochs) as u64);
        let n_nodes = self.n_hidden_nodes[i];
        let n_dropped_nodes =
            (self.dropout_rates[i] * self.n_hidden_nodes[i] as f32).round() as usize;
        let weights_x_dropout = if n_dropped_nodes > 0 {
            let idx_dropped_nodes = (0..n_nodes).choose_multiple(&mut rng, n_dropped_nodes);
            let mut d = vec![1.0f32; n_nodes];
            for i in idx_dropped_nodes {
                d[i] = 0.0;
            }
            let d_dev: CudaSlice<f32> = self
                .targets
                .data
                .context()
                .default_stream()
                .clone_htod(&d)?;
            let d_matrix = Matrix::new(d_dev, n_nodes, 1)?;
            let x = self.weights_per_layer[i].rowmatmul(&d_matrix)?;
            x.matmul(&self.activations_per_layer[i])?
        } else {
            self.weights_per_layer[i].matmul(&self.activations_per_layer[i])?
        };
        self.weights_x_biases_per_layer[i] =
            weights_x_dropout.rowmatadd(&self.biases_per_layer[i])?;
        Ok(())
    }
}

// TODO: add tests