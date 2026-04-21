use crate::network::Network;
use crate::optimisers::OptimisationParameters;
use crate::train::TrainingError;
use chrono::Utc;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;

use std::env::current_dir;
use std::error::Error;
use std::path::PathBuf;
use std::cmp::Ordering;

impl From<PlottingError> for TrainingError {
    fn from(err: PlottingError) -> Self {
        TrainingError::OtherError(err.to_string())
    }
}

impl Network {
    pub fn plot_loss(self: &Self, epochs: Vec<Vec<f64>>, costs: Vec<Vec<f64>>, optimisation_parameters: &OptimisationParameters) -> Result<String, Box<dyn Error>> {
        // Filename
        let dir: PathBuf = current_dir()?;
        let fname_loss_svg = format!(
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
        // Plot loss curve
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
        plot_loss[0].clone().export_svg(&fname_loss_svg)?;
        Ok(fname_loss_svg)
    }

    pub fn plot_true_vs_pred(self: &Self, optimisation_parameters: &OptimisationParameters) -> Result<String, Box<dyn Error>> {
        // Filename
        let dir: PathBuf = current_dir()?;
        let fname_scatter_svg = format!(
            "{}/Observed_vs_predicted-HL{}-{:?}-{:?}-E{}-FPE{}-B{}-LR{}-T{}.svg",
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
            .export_svg(&fname_scatter_svg)?;
        Ok(fname_scatter_svg)
    }
}

// impl Marginals {
//     pub fn plot
// }

#[cfg(test)]
mod test {
    use super::*;
    use crate::io::Data;
    use rand::Rng;
    #[test]
    fn test_plot() -> Result<(), Box<dyn Error>> {
        let n: usize = 12_345; // number of observations
        let p: usize = 17; // number of input features
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let data = Data::simulate(n, p, k, n_hidden_layers, "normal", 0.0, 1.0, 42)?;
        let network = data.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
        let optimisation_parameters = OptimisationParameters::new(&network)?;

        let mut rng = rand::rng();
        let m: usize = 3;
        let q: usize = 100;
        let mut epochs: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut costs: Vec<Vec<f64>> = Vec::with_capacity(m);
        for _ in 0..m {
            let mut e: Vec<f64> = Vec::with_capacity(q);
            let mut c: Vec<f64> = Vec::with_capacity(q);
            for j in 0..q {
                e.push(j as f64);
                c.push(rng.random::<f64>());
            }
            epochs.push(e);
            costs.push(c);
        }
        let fname_loss = network.plot_loss(epochs, costs, &optimisation_parameters)?;
        let fname_true_vs_pred = network.plot_true_vs_pred(&optimisation_parameters)?;
        // Clean-up
        std::fs::remove_file(&fname_loss)?;
        std::fs::remove_file(&fname_true_vs_pred)?;
        Ok(())
    }
}