use std::io::{stdout, Write};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ProgressBar {
    start_time: Instant,
    counter: usize,
    total: usize,
    progress_width: usize,
    title: String,
}

impl ProgressBar {
    pub fn new(total: usize, progress_width: usize, title: String) -> Self {
        ProgressBar {
            start_time: Instant::now(),
            counter: 0,
            total,
            progress_width,
            title,
        }
        
    }

    pub fn next(self: &mut Self) -> () {
        let perc: f64 = (((self.progress_width * 100 * (self.counter+1)) as f64)/(self.total as f64)).round() / (self.progress_width as f64);
        let n_progress: usize = (((self.progress_width * (self.counter+1)) as f64) / (self.total as f64)).round() as usize;
        let progress_text: String = (0..n_progress).map(|_| "█").collect();
        let no_progress_text: String = (0..(self.progress_width-n_progress)).map(|_| " ").collect();
        let t_remaining: f64 = {
            let dp: f64 = n_progress as f64;
            let dt: f64 = self.start_time.elapsed().as_millis() as f64 / 60_000.0;
            let v: f64 = dp/dt;
            let t_total: f64 = (self.progress_width as f64) / v;
            t_total - dt
        };
        print!("\r{} | {:.2}% | {}{} | {:.2} minutes remaining | ", self.title, perc, progress_text, no_progress_text, t_remaining);
        stdout().flush().expect("Failed to flush stdout");
        // Increment the counter
        self.counter += 1;
    }

    pub fn finish(self: &mut Self) -> () {
        let progress_text: String = (0..self.progress_width).map(|_| "█").collect();
        print!("\r{} | 100.00% | {} |", self.title, progress_text);
        stdout().flush().expect("Failed to flush stdout");
        println!(" Duration: {:.2} minutes", self.start_time.elapsed().as_millis() as f64 / 60_000.0);
    }
}
