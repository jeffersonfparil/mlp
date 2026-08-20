use crate::linalg::matrix::Matrix;
use crate::network::Network;
use cudarc::driver::CudaSlice;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use std::error::Error;

impl Network {
    pub fn generate_dropout_mask(&mut self, i: usize) -> Result<(), Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64((self.seed + i + self.n_epochs) as u64);
        let n_nodes = self.n_hidden_nodes[i];
        let n_dropped_nodes =
            (self.dropout_rates[i] * self.n_hidden_nodes[i] as f32).round() as usize;
        if n_dropped_nodes > 0 {
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
            self.dropout_masks_per_layer[i] = Matrix::new(d_dev, n_nodes, 1)?;
        };
        Ok(())
    }

    pub fn dropout(&mut self, i: usize) -> Result<(), Box<dyn Error>> {
        self.weights_x_biases_per_layer[i] = self.weights_per_layer[i]
            .rowmatmul(&self.dropout_masks_per_layer[i])?
            .matmul(&self.activations_per_layer[i])?
            .rowmatadd(&self.biases_per_layer[i])?;
        Ok(())
    }

    pub fn drop_dropout_mask(&mut self, i: usize) -> Result<(), Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64((self.seed + i + self.n_epochs) as u64);
        let n_nodes = self.n_hidden_nodes[i];
        let n_dropped_nodes =
            (self.dropout_rates[i] * self.n_hidden_nodes[i] as f32).round() as usize;
        let d = vec![1.0f32; n_nodes];
        let d_dev: CudaSlice<f32> = self
            .targets
            .data
            .context()
            .default_stream()
            .clone_htod(&d)?;
        self.dropout_masks_per_layer[i] = Matrix::new(d_dev, n_nodes, 1)?;
        Ok(())
    }
}

// TODO: add tests