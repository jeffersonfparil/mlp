use crate::network::Network;
use crate::marginal::Marginals;
use crate::optimisers::OptimisationParameters;
use crate::train::TrainingError;
use chrono::Utc;
use ruviz::core::{Plot, PlottingError, SubplotFigure, subplots};
use ruviz::prelude::{LegendPosition, PlotBuilder};
use ruviz::plots::BarConfig;
use std::env::current_dir;
use std::error::Error;
use std::path::PathBuf;
use std::cmp::Ordering;

impl From<PlottingError> for TrainingError {
    fn from(err: PlottingError) -> Self {
        TrainingError::OtherError(err.to_string())
    }
}

// NOTE: The following methods generate quick and dirty plots NOT meant as final or production-grade or publication-quality plots.

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

impl Marginals {

    pub fn plot(self: &Self, main_only: bool) -> Result<String, Box<dyn Error>> {
        // Note that we only plot the marginal effects and leave it to the use to plot the R2 using their preferred plotting software/library
        // Filename
        let dir: PathBuf = current_dir()?;
        let fname_main_effects_png = format!(
            "{}/Marginal_effects-T{}.png",
            dir.display(),
            Utc::now().format("%Y%m%d%H%M%S")
        );
        // Extract ids and effects
        let mut ids_all: Vec<String> = Vec::new();
        let mut effects_all: Vec<f64> = Vec::new();
        for i in 0..self.ids.len() {
            let id = self.ids[i].to_owned();
            let id_split = id.split("▓").collect::<Vec<&str>>();
            if main_only && (id_split.len() > 1) {
                continue;
            }
            ids_all.push(id);
            effects_all.push(self.effects[i] as f64);
        }
        let (max_x_tick_labels, layout_n_plots) = {
            let max_x_tick_labels: usize = 10;
            let layout_n_plots: usize = ids_all.len().div_ceil(max_x_tick_labels);
            if layout_n_plots > 10 {
                (10, ids_all.len().div_ceil(layout_n_plots))
            } else {
                (max_x_tick_labels, layout_n_plots)
            }
        };
        // TODO: Post an issue in ruviz: using `ylim(min, max)` does not work!
        // let y_min = match effects_all.iter().min_by(|&a, &b| a.partial_cmp(b).unwrap()) {
        //     Some(x) => *x,
        //     None => f64::NEG_INFINITY,
        // };
        // let y_max = match effects_all.iter().max_by(|&a, &b| a.partial_cmp(b).unwrap()) {
        //     Some(x) => *x,
        //     None => f64::INFINITY,
        // };
        // println!("y_min={}; y_max={}", y_min, y_max);
        let mut plots: Vec<PlotBuilder<BarConfig>> = Vec::with_capacity(layout_n_plots);
        let mut ini: usize = 0;
        for _ in 0..layout_n_plots {
            let fin: usize = if (ini+max_x_tick_labels) < ids_all.len() {
                ini+max_x_tick_labels
            } else {
                ids_all.len()
            };
            // let ids: Vec<String> = ids_all[ini..fin].to_vec();
            let ids: Vec<String> = ids_all[ini..fin].iter().map(|x| x.replace("▓", "\n")).collect();
            let effects: Vec<f64> = effects_all[ini..fin].to_vec();
            let title = if main_only {
                format!("Main Marginal Effects ({}/{})", ids.len(), ids_all.len())
            } else {
                format!("All Marginal Effects ({}/{})", ids.len(), ids_all.len())
            };
            let plot = Plot::new()
                    .title(title.as_str())
                    .ylabel("Effect")
                    // .ylim(y_min, y_max)
                    .bar(&ids, &effects);
            plots.push(plot);
            // println!("ini={}; fin={}; effects: {:?}", ini, fin, effects);
            ini = fin;
        }
        let mut plot = vec![SubplotFigure::new(layout_n_plots, 1, 1_200, (layout_n_plots * 300) as u32)?];
        for i in 0..layout_n_plots {
            plot[0] = plot[0].clone().subplot(i, 0, plots[i].clone().into())?.clone();
        }
        plot[0].clone().save(&fname_main_effects_png)?;
        // plot[0].clone().save_with_dpi(&fname_main_effects_png, 300.0)?;
        // plot[0].clone().export_svg(&fname_main_effects_png)?;
        Ok(fname_main_effects_png)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::io::Data;
    use rand::Rng;
    #[test]
    fn test_plot() -> Result<(), Box<dyn Error>> {
        let n: usize = 12_345; // number of observations
        let p: usize = 17; // number of input features (continuous)
        let q: Vec<usize> = vec![3,4,5,10]; // number of input features (categorical levels)
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        // We use half the number of input features as the number of nodes in the hidden layers, i.e. let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        // let data = Data::new(100, 10, 1)?; // Just a bunch of zeros
        let data = Data::simulate(n, p, q, k, n_hidden_layers, "normal", 0.0, 1.0, 42, true)?;
        let network = data.init_network(2, vec![5; 2], vec![0.0; 2], 42)?;
        let optimisation_parameters = OptimisationParameters::new(&network)?;
        // Network-related plots
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
        let _fname_loss = network.plot_loss(epochs, costs, &optimisation_parameters)?;
        let _fname_true_vs_pred = network.plot_true_vs_pred(&optimisation_parameters)?;
        // Marginal effects-related plots
        let mut marginals = Marginals::new(data.feature_names, 3)?;
        for i in 0..marginals.ids.len() {
            marginals.effects[i] = rng.random::<f32>();
        }
        // println!("marginals: {:?}", marginals);
        let _fname_main_effects_png = marginals.plot(true)?;
        let _fname_all_effects_png = marginals.plot(false)?;
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