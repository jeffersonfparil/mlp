use crate::{linalg::matrix::MatrixError, network::Network};
use std::error::Error;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;
use std::cmp::Ordering;
use itertools::Itertools;


#[derive(Debug, Clone)]
pub struct Marginals {
    pub ids: Vec<String>,
    pub effects: Vec<f32>,
}

impl Marginals {
    pub fn new(feature_names: Vec<String>, order: usize) -> Result<Self, Box<dyn Error>> {
        let n = feature_names.len();
        let mut p = 0;
        for i in 1..=order {
            p += n.pow(i as u32);
        }
        let mut ids: Vec<String> = vec![];
        for i in 1..=order {
            'combi: for combination in (0..n).into_iter().combinations(i) {
                // Skip we have duplicate features in a combination
                // Skip if we have non-numeric features if they are the same feature just different levels
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
        let marginals  = Marginals {
            ids,
            effects: vec![f32::NAN; p],
        };
        Ok(marginals)
    }

    pub fn check_dimensions(&self) -> Result<(), MatrixError> {
        if self.ids.len() != self.effects.len() {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of ids ({}) does not match number of effects ({}).",
                self.ids.len(), self.effects.len()
            )));
        }
        Ok(())
    }

    pub fn estimate_effects(self: &mut Self, network: &mut Network, m: usize, verbose: bool) -> Result<(), Box<dyn Error>> {

        // Find the range of values for each input node
        println!("number of activation layers: {}", network.activations_per_layer.len());
        println!("input_layer: {}", network.activations_per_layer[0]);

        let n: usize = network.activations_per_layer[0].n_cols;
        let p: usize = network.activations_per_layer[0].n_rows;

        // let mut minima: Vec<f32> = vec![f32::NAN; p];
        // let mut maxima: Vec<f32> = vec![f32::NAN; p];
        let input_matrix_orig = network.activations_per_layer[0].to_host()?;
        let mut input_matrix = input_matrix_orig.clone();
        let stream = network.activations_per_layer[0].data.context().default_stream();
        
        let mut feature_names: Vec<String> = vec![];
        for i in 0..self.ids.len() {
            let id = self.ids[i].to_owned();
            let id_split = id.split("▓").into_iter().map(|x| x.to_owned()).collect::<Vec<String>>();
            if id_split.len() == 1 {
                feature_names.push(id);
            }
        }

        // Emit custom MarginalError here
        assert_eq!(p, feature_names.len());

        // Define the ranges for each of the features
        let mut ranges: Vec<Vec<f32>> = vec![];
        for j in 0..p {
            let ini: usize = j * n;
            let fin: usize = (j + 1) * n;
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
            ranges.push(new_values_to_iterate);
        }
        println!("ranges: {:?}", ranges);
        

        for i in 0..self.ids.len() {
            let id = self.ids[i].to_owned();
            let id_split = id.split("▓").into_iter().map(|x| x.to_owned()).collect::<Vec<String>>();
            let mut idx_split: Vec<usize> = vec![];
            for j in 0..id_split.len() {
                let x = id_split[j].to_owned();
                let y = feature_names
                    .iter()
                    .enumerate()
                    .filter(move |&(j, z)| &x == z)
                    .map(|(k, _)| k)
                    .collect::<Vec<usize>>();
                if y.len() != 1 {
                    return Err(Box::new(MatrixError::OtherError(format!("Unrecognised feature name: `{}`", id_split[j].to_owned()))))
                }
                idx_split.push(y[0]);
            }
            println!("idx_split: {:?}", idx_split);

            if idx_split.len() == 1 {

                // Let's first do for no interactions

                let idx = idx_split[0];
                let ini: usize = idx * n;
                let fin: usize = (idx + 1) * n;
                let mut x: Vec<f64> = vec![f64::NAN; m*n]; // new input values
                let mut y: Vec<f64> = vec![f64::NAN; m*n]; // resulting changes to predictions
                for (j, x_i) in ranges[idx].clone().into_iter().enumerate() {
                    for k in 0..n {
                        input_matrix[ini+k] = x_i;
                    }
                    network.activations_per_layer[0].data = stream.clone_htod(&input_matrix)?;
                    network.predict()?;
                    let predictions = network.predictions.to_host()?;
                    for k in 0..n {
                        x[(j*n)+k] = x_i as f64;
                        y[(j*n)+k] = predictions[k] as f64;
                    }
                }
                let b: f64 = {
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
                    cov_xy / (var_x + epsilon)
                };
                self.effects[i] = b as f32;
                // Reset the network to previous state
                network.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
                network.predict()?;

            } else {

                // TODO: estimate the marginal effects where we simplistically assume increase in interaction effects to be along the same order for now

            }


            // for j in idx_split {
            //     let ini: usize = j * n;
            //     let fin: usize = (j + 1) * n;
            //     let old_values = match input_matrix.get(ini..fin) {
            //         Some(x) => x.to_owned(),
            //         None => return Err(Box::new(MatrixError::DimensionMismatch(format!("Inappropriate slicing index from {} to {}.", ini, fin)))),
            //     };
            //     // println!("old_values[0]={}, old_values[1]={}, old_values[2]={}, old_values[3]={}", old_values[0], old_values[1], old_values[2], old_values[3]);
            //     let min = match old_values.iter().filter(|&a| !a.is_nan()).min_by(|&a, &b| a.total_cmp(b)) {
            //         Some(&a) => a,
            //         None => f32::NAN,
            //     };
            //     let max = match old_values.iter().filter(|&a| !a.is_nan()).max_by(|&a, &b| a.total_cmp(b)) {
            //         Some(&a) => a,
            //         None => f32::NAN,
            //     };
            //     let step_size = (max - min) / ((m-1) as f32);
            //     let new_values_to_iterate: Vec<f32> = (0..m).map(|x| min+(step_size*(x as f32))).collect();
            //     let mut x: Vec<f64> = vec![f64::NAN; m*n]; // new input values
            //     let mut y: Vec<f64> = vec![f64::NAN; m*n]; // resulting changes to predictions

            //     println!("min={}; max={}; step_size={}; new_values_to_iterate={:?}, m={}; n={}; m*n={}", min, max, step_size, new_values_to_iterate, m, n, m*n);

            //     for k in 0..m {
            //         let v = new_values_to_iterate[k];
            //         for k in 0..n {
            //             input_matrix[ini+k] = v;
            //         }
            //         network.activations_per_layer[0].data = stream.clone_htod(&input_matrix)?;
            //         network.predict()?;
            //         let predictions = network.predictions.to_host()?;
            //         for l in 0..n {
            //             x[(k*n)+l] = v as f64;
            //             y[(k*n)+l] = predictions[l] as f64;
            //         }
            //     }
            //     // println!("x[0]={}, x[1]={}, x[2]={}, x[3]={}", x[0], x[1], x[2], x[3]);
            //     // println!("y[0]={}, y[1]={}, y[2]={}, y[3]={}", y[0], y[1], y[2], y[3]);
            //     let epsilon: f64 = 1e-7;
            //     let n: f64 = x.len() as f64;
            //     let u_x: f64 = x.iter().fold(0.0, |sum, x| sum + x) / n;
            //     let u_y: f64 = y.iter().fold(0.0, |sum, x| sum + x) / n;
            //     let cov_xy: f64 = x
            //         .iter()
            //         .zip(y.iter())
            //         .fold(0.0, |a, (x, y)| a + (x - u_x) * (y - u_y));
            //     let var_x: f64 = x
            //         .iter()
            //         .fold(0.0, |a, x| a + (x - u_x).powi(2));
            //     let var_y: f64 = y
            //         .iter()
            //         .fold(0.0, |a, x| a + (x - u_y).powi(2));
            //     let b: f64 = cov_xy / (var_x + epsilon);
            //     let a: f64 = u_y - (b * u_x);
            //     if verbose {
            //         let error2: f64 = y
            //             .iter()
            //             .zip(x.iter())
            //             .fold(0.0, |a, (y_p, y_t)| a + (y_p - y_t).powi(2));
            //         let r2: f64 = 1.00 - (error2 / (var_x + epsilon));
            //         let cor: f64 = cov_xy / ((var_x.sqrt() * var_y.sqrt()) + epsilon);
            //         let x_min: f64 = match x
            //             .iter()
            //             .min_by(|a, b| {
            //                 a.partial_cmp(b).unwrap_or(Ordering::Less)
            //             }) {
            //                 Some(x) => *x,
            //                 None => x[0],
            //             };
            //         let x_max: f64 = match x
            //             .iter()
            //             .max_by(|a, b| {
            //                 a.partial_cmp(b).unwrap_or(Ordering::Greater)
            //             }) {
            //                 Some(x) => *x,
            //                 None => x[0],
            //             };
            //         let mut x_for_plotting: Vec<f64> = Vec::with_capacity(100);
            //         let mut y_for_plotting: Vec<f64> = Vec::with_capacity(100);
            //         let step: f64 = (x_max - x_min) / 99.0;
            //         for k in 0..100 {
            //             let x: f64 = x_min + (k as f64 * step);
            //             let y: f64 = a + x*b;
            //             x_for_plotting.push(x);
            //             y_for_plotting.push(y);
            //         }
            //         let title = format!("Sensitivity values vs Predicted Values\n(b={}; R2={:.2}; cor={:.2})", b, r2, cor);
            //         Plot::new()
            //             .title(title.as_str())
            //             .xlabel("x")
            //             .ylabel("y")
            //             .scatter(&x, &y)
            //             .line(&x_for_plotting, &y_for_plotting)
            //             .export_svg(format!("test-{}.svg", j))?;
            //     }
            //     self.effects[j] = b as f32;
            //     // Reset the network to previous state
            //     network.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
            //     network.predict()?;
            // }
        }
        // // Reset the network to previous state
        // network.activations_per_layer[0].data = stream.clone_htod(&input_matrix_orig)?;
        // network.predict()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Data;
    use crate::optimisers::OptimisationParameters;
    #[test]
    fn test_marginal() -> Result<(), Box<dyn Error>> {
 
        let feature_names: Vec<String> = vec!["feature_0".to_owned(), "feature_1".to_owned(),"feature_A".to_owned(),"feature_B".to_owned()];
        let marginals_order_1 = Marginals::new(feature_names.clone(), 1)?;
        println!("marginals_order_1: {:?}", marginals_order_1);
        let marginals_order_2 = Marginals::new(feature_names.clone(), 2)?;
        println!("marginals_order_2: {:?}", marginals_order_2);
        let marginals_order_3 = Marginals::new(feature_names.clone(), 3)?;
        println!("marginals_order_3: {:?}", marginals_order_3);
 
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
        
        let mut marginals = Marginals::new(data.feature_names.clone(), 1)?;
        let number_of_values_for_interpolate_between_min_and_max: usize = 10;
        marginals.estimate_effects(&mut network, number_of_values_for_interpolate_between_min_and_max, true)?;
        println!("marginals: {:?}", marginals);


        Ok(())
    }
}