use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::error::Error;
use std::sync::Arc;

/// Stores the result of a fitted PCA model.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Eigen {
    /// The Principal Components (eigenvectors) of shape (n_cols x num_pcs)
    pub loadings: Matrix,
    /// The true sample eigenvalues corresponding to each Principal Component
    pub eigenvalues: Vec<f32>,
    /// The proportion of variance explained by each Principal Component (0.0 to 1.0)
    pub variance_explained: Vec<f32>,
    /// The negative column means used for centering, saved for projecting new data
    pub col_means_neg: Matrix,
}

impl Eigen {
    /// Projects a new set of observations onto the extracted Principal Components.
    /// Returns a new Matrix of size (new_data.n_rows x num_pcs).
    pub fn transform(&self, new_data: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        let centered_data = new_data.colmatadd(&self.col_means_neg)?;
        let projected = centered_data.matmul(&self.loadings)?;
        Ok(projected)
    }
}

impl Matrix {
    /// Computes up to `num_pcs` Principal Components using Power Iteration 
    /// and Hotelling's Deflation. 
    pub fn principal_components(
        &self, 
        num_pcs: usize, 
        max_iter: usize, 
        tol: f32
    ) -> Result<Eigen, Box<dyn Error>> {
        let stream = self.data.context().default_stream();
        let k = num_pcs.min(self.n_cols);
        
        // Centre the data. 
        // Note: Assuming colsummat() returns column sums, we multiply by -1/N to get the negative mean.
        let col_means_neg = self.colsummat()?.scalarmatmul(-1.0 / (self.n_rows as f32))?;
        println!("col_means_neg: {}", col_means_neg);
        
        let centered_mat = self.colmatadd(&col_means_neg)?;
        println!("centered_mat: {}", centered_mat);
        
        // Scatter Matrix (X^T * X) - Unscaled Covariance
        let mut cov_mat = centered_mat.matmult0(&centered_mat)?;
        
        // Calculate Total Variance (Trace of X^T * X) for explained variance ratios.
        // The sum of squared elements of the centered matrix equals the trace of X^T * X.
        let total_variance_unscaled = centered_mat
            .elementwisematmul(&centered_mat)?
            .summat()?;

        // Vectors to hold our new output metrics
        let mut loadings_host = vec![0.0f32; self.n_cols * k];
        let mut eigenvalues = Vec::with_capacity(k);
        let mut variance_explained = Vec::with_capacity(k);
        
        // Degrees of freedom for sample covariance
        let degrees_of_freedom = (self.n_rows as f32 - 1.0).max(1.0);

        // Extract PCs sequentially
        for pc_idx in 0..k {
            let v_mat = {
                let mut v_mat = Matrix::new(stream.clone_htod(&vec![1.0f32; self.n_cols])?, self.n_cols, 1)?;
                for _ in 0..max_iter {
                    let v_new_mat = cov_mat.matmul(&v_mat)?;
                    let norm = v_new_mat
                        .elementwisematmul(&v_new_mat)?
                        .summat()?
                        .sqrt();
                        
                    v_mat = v_new_mat.scalarmatmul(1.00 / norm)?;
                    let diff = v_mat
                        .elementwisematadd(
                            &v_mat.scalarmatmul(-1.00)?
                        )?
                        .elementwisematabs()?
                        .summat()?;
                        
                    if diff < tol { 
                        break; 
                    }
                }
                v_mat
            };
            
            // Unscaled Eigenvalue (Variance) Calculation: lambda_unscaled = v^T * C * v
            let lambda_unscaled = v_mat
                .matmult0(&cov_mat)?
                .matmul(&v_mat)?
                .to_host()?[0];
                
            // 1. Calculate true Sample Covariance Eigenvalue
            let lambda = lambda_unscaled / degrees_of_freedom;
            eigenvalues.push(lambda);
            
            // 2. Calculate Percentage of Variance Explained
            let pct_explained = if total_variance_unscaled > 0.0 {
                lambda_unscaled / total_variance_unscaled
            } else {
                0.0
            };
            variance_explained.push(pct_explained);
            // Store the Eigenvector into the host matrix
            let v_host = v_mat.to_host()?;
            for i in 0..self.n_cols {
                loadings_host[i * k + pc_idx] = v_host[i];
            }
            // Hotelling's Deflation (C = C - lambda_unscaled * v * v^T)
            // Note: We use the unscaled lambda here because `cov_mat` is X^T X, not X^T X / (N-1)
            if pc_idx < k - 1 {
                cov_mat = cov_mat.elementwisematadd(
                    &v_mat
                        .matmul0t(&v_mat)?
                        .scalarmatmul(-lambda_unscaled)?
                )?;
            }
        }
        // Push the final loadings matrix to the GPU
        let loadings = Matrix::new(stream.clone_htod(&loadings_host)?, self.n_cols, k)?;
        Ok(Eigen {
            loadings,
            eigenvalues,
            variance_explained,
            col_means_neg,
        }) 
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::safe::CudaContext;
    use cudarc::driver::CudaSlice;
    use std::error::Error;

    #[test]
    fn test_eigen() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let n: usize = 10;
        let p: usize = 7;
        let k: usize = 3;
        let mut x_host: Vec<f32> = (0..(n * p)).map(|x| 1.1_f32.powf(x as f32)).collect();
        for i in (0..(n*p)).step_by(10) {
            x_host[i] = (n*p - i) as f32;
        }
        let x_dev: CudaSlice<f32> = stream.clone_htod(&x_host)?;
        let x_matrix = Matrix::new(x_dev, n, p)?;

        // 2. Calculate the first Principal Component
        // Using max_iter = 100 and tol = 1e-5
        let pca_result = x_matrix.principal_components(k, 100, 1e-5)?;

        println!("pca_result: {:?}", pca_result);
        println!("x_matrix: {}", x_matrix);
        println!("loadings: {}", pca_result.loadings);
        println!("eigenvalues: {:?}", pca_result.eigenvalues);
        println!("variance explained: {:?}", pca_result.variance_explained);

        // 3. Fetch the loadings matrix back to the host
        let result_pc1: Vec<f32> = pca_result.loadings.to_host()?;
        let expected_pc1: Vec<f32> = vec![0.2739837, 0.3013821, 0.3315203, 0.3646723, 0.4011395, 0.4412535, 0.4853788];
        
        println!("result_pc1: {:?}", result_pc1);
        println!("expected_pc1: {:?}", expected_pc1);
        
        for i in 0..p {
            assert!((result_pc1[i] - expected_pc1[i]).abs() < 0.001);
        }

        // 4. Test transformation of new data (using the same data to verify shape)
        let transformed = pca_result.transform(&x_matrix)?;

        println!("x_matrix: {}", x_matrix);
        println!("transformed: {}", transformed);

        assert_eq!(transformed.n_rows, n);
        assert_eq!(transformed.n_cols, 1);

        Ok(())
    }
}