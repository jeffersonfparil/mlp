use crate::{linalg::matrix::MatrixError, network::Network};
use std::error::Error;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;
use std::cmp::Ordering;


// Extract marginal effects of each input node, i.e. each explanatory variable and their levels if any
impl Network {
    pub fn marginals(self: &mut Self, m: usize, verbose: bool) -> Result<Vec<f32>, Box<dyn Error>> {

        // Find the range of values for each input node
        println!("number of activation layers: {}", self.activations_per_layer.len());
        println!("input_layer: {}", self.activations_per_layer[0]);

        let n: usize = self.activations_per_layer[0].n_cols;
        let p: usize = self.activations_per_layer[0].n_rows;

        // let mut minima: Vec<f32> = vec![f32::NAN; p];
        // let mut maxima: Vec<f32> = vec![f32::NAN; p];
        let input_matrix_orig = self.activations_per_layer[0].to_host()?;
        let mut input_matrix = input_matrix_orig.clone();
        let stream = self.activations_per_layer[0].data.context().default_stream();
        
        // TODO: include interaction effects where highest degree equals the number of hidden layers
        
        let mut effects: Vec<f32> = vec![f32::NAN; p];
        for i in 0..p {
            // let old_values = self.activations_per_layer[0]
            //     .slice(&vec![i], &(0..n).collect())?
            //     .to_host()?;
            let ini: usize = i * n;
            let fin: usize = (i + 1) * n;
            let old_values = match input_matrix.get(ini..fin) {
                Some(x) => x.to_owned(),
                None => return Err(Box::new(MatrixError::DimensionMismatch(format!("Inappropriate slicing index from {} to {}.", ini, fin)))),
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
            let step_size = (max - min) / ((m-1) as f32);
            let new_values_to_iterate: Vec<f32> = (0..m).map(|x| min+(step_size*(x as f32))).collect();
            let mut x: Vec<f64> = vec![f64::NAN; m*n]; // new input values
            let mut y: Vec<f64> = vec![f64::NAN; m*n]; // resulting changes to predictions
            for j in 0..m {
                let v = new_values_to_iterate[j];
                for k in 0..n {
                    input_matrix[ini+k] = v;
                }
                self.activations_per_layer[0].data = stream.clone_htod(&input_matrix)?;
                self.predict()?;
                let predictions = self.predictions.to_host()?;
                for k in 0..n {
                    x[(j*n)+k] = v as f64;
                    y[(j*n)+k] = predictions[k] as f64;
                }
            }
            // println!("x[0]={}, x[1]={}, x[2]={}, x[3]={}", x[0], x[1], x[2], x[3]);
            // println!("y[0]={}, y[1]={}, y[2]={}, y[3]={}", y[0], y[1], y[2], y[3]);

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
            let var_y: f64 = y
                .iter()
                .fold(0.0, |a, x| a + (x - u_y).powi(2));
            let b: f64 = cov_xy / (var_x + epsilon);
            let a: f64 = u_y - (b * u_x);
            if verbose {
                let error2: f64 = y
                    .iter()
                    .zip(x.iter())
                    .fold(0.0, |a, (y_p, y_t)| a + (y_p - y_t).powi(2));
                let r2: f64 = 1.00 - (error2 / (var_x + epsilon));
                let cor: f64 = cov_xy / ((var_x.sqrt() * var_y.sqrt()) + epsilon);
                let x_min: f64 = match x
                    .iter()
                    .min_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(Ordering::Less)
                    }) {
                        Some(x) => *x,
                        None => x[0],
                    };
                let x_max: f64 = match x
                    .iter()
                    .max_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(Ordering::Greater)
                    }) {
                        Some(x) => *x,
                        None => x[0],
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
                let title = format!("Sensitivity values vs Predicted Values\n(b={}; R2={:.2}; cor={:.2})", b, r2, cor);
                Plot::new()
                    .title(title.as_str())
                    .xlabel("x")
                    .ylabel("y")
                    .scatter(&x, &y)
                    .line(&x_for_plotting, &y_for_plotting)
                    .export_svg(format!("test-{}.svg", i))?;
            }

            effects[i] = b as f32;

            // minima[i] = min;
            // maxima[i] = max;
        }
        // println!("minima: {:?}", minima);
        // println!("maxima: {:?}", maxima);
        // println!("effects: {:?}", effects);

        // Reset the network to previous state
        self.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
        self.predict()?;
     

        Ok(effects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Data;
    use crate::optimisers::OptimisationParameters;
    #[test]
    fn test_marginal() -> Result<(), Box<dyn Error>> {
        let n: usize = 50; // number of observations
        let p: usize = 7; // number of input features
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let data = Data::simulate(n, p, k, n_hidden_layers, "normal", 0.0, 1.0, 42)?;
        let mut network = data.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        network.train(&mut optimisation_parameters, true)?;
        
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        let effects = network.marginals(number_of_values_for_interpolate_between_min_and_max, false)?;
        println!("effects: {:?}", effects);


        Ok(())
    }
}