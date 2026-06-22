use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const DEFLATE: &str = "
    extern \"C\" __global__ void cuDeflate(float* C, float* v, float lambda, int n_cols) {
        // Hotelling's Deflation kernel implementation: C = C - lambda * v * v^T
        // Arguments:
        //  - C: Covariance matrix (n_cols x n_cols) to be updated in-place
        //  - v: Eigenvector (n_cols x 1)
        //  - lambda: Eigenvalue corresponding to v
        //  - n_cols: Number of columns in the original dataset
        
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        
        if ((i < n_cols) && (j < n_cols)) {
            C[i * n_cols + j] -= lambda * v[i] * v[j];
        }
    }
";

impl Matrix {
    /// Performs Hotelling's Deflation in-place: C = C - lambda * v * v^T
    /// This removes the variance explained by the given principal component 
    /// from the covariance matrix, allowing the extraction of the next component.
    pub fn deflate_mut(
        &mut self, 
        v: &Matrix, 
        lambda: f32
    ) -> Result<(), Box<dyn Error>> {
        // Validation checks
        if self.n_rows != self.n_cols {
            return Err(Box::new(MatrixError::DimensionMismatch(
                format!("Deflation requires a square covariance matrix, got {}x{}", self.n_rows, self.n_cols)
            )));
        }
        if v.n_rows != self.n_cols || v.n_cols != 1 {
            return Err(Box::new(MatrixError::DimensionMismatch(
                format!("Eigenvector must be a column vector of size {}x1, got {}x{}", self.n_cols, v.n_rows, v.n_cols)
            )));
        }

        // Fetch the cached kernel
        let f: CudaFunction = self.get_cached_kernel("cuDeflate", DEFLATE)?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        // let cfg = LaunchConfig {
        //     grid_dim: (
        //         (self.n_cols as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        //         (self.n_cols as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        //         1,
        //     ),
        //     block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        //     shared_mem_bytes: 0,
        // };
        // // Launch the kernel, passing mutable reference to self.data
        // unsafe {
        //     f.launch(
        //         cfg,
        //         (
        //             &mut self.data, 
        //             &v.data, 
        //             lambda, 
        //             self.n_cols as i32
        //         )
        //     )?;
        // }
        let n_rows: u32 = self.n_cols as u32;
        let n_cols: u32 = self.n_rows as u32;
        
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        builder.arg(&mut self.data);
        builder.arg(&v.data);
        builder.arg(&lambda);
        builder.arg(&self.n_cols);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
            grid_dim: (
                (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1,
            ),
            shared_mem_bytes: 0,
        };
        unsafe {
            let _ = builder.launch(cfg);
        };

        Ok(())
    }

    /// Computes up to `num_pcs` Principal Components (loadings) using Power Iteration 
    /// and Hotelling's Deflation. 
    /// Returns a new Matrix of size (n_cols x num_pcs).
    pub fn principal_components(
        &self, 
        num_pcs: usize, 
        max_iter: usize, 
        tol: f32
    ) -> Result<Matrix, Box<dyn Error>> {
        let stream = self.data.context().default_stream();

        // Cap the number of PCs to the number of columns available
        let k = num_pcs.min(self.n_cols);

        // 1. Center the Data (Host-side O(N) operation)
        let host_data = self.to_host()?;
        let mut col_means = vec![0.0f32; self.n_cols];
        
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                col_means[j] += host_data[i * self.n_cols + j];
            }
        }
        for j in 0..self.n_cols {
            col_means[j] /= self.n_rows as f32;
        }

        let mut centered_host = vec![0.0f32; host_data.len()];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                centered_host[i * self.n_cols + j] = host_data[i * self.n_cols + j] - col_means[j];
            }
        }
        let centered_dev = stream.clone_htod(&centered_host)?;
        let centered_mat = Matrix::new(centered_dev, self.n_rows, self.n_cols)?;

        // 2. Compute Initial Covariance Matrix (X^T * X)
        let mut cov_mat = centered_mat.matmult0(&centered_mat)?;

        // Allocate host memory to store the final combined loadings matrix (row-major)
        let mut loadings_host = vec![0.0f32; self.n_cols * k];

        // 3. Extract PCs sequentially
        for pc_idx in 0..k {
            let mut v_host = vec![1.0f32; self.n_cols];
            let mut v_mat = Matrix::new(stream.clone_htod(&v_host)?, self.n_cols, 1)?;

            // --- Power Iteration Loop ---
            for _ in 0..max_iter {
                let v_new_mat = cov_mat.matmul(&v_mat)?;
                let v_new_host = v_new_mat.to_host()?;
                
                let norm = v_new_host.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mut diff = 0.0;
                
                for i in 0..self.n_cols {
                    let normalized = v_new_host[i] / norm;
                    diff += (normalized - v_host[i]).abs();
                    v_host[i] = normalized;
                }

                v_mat = Matrix::new(stream.clone_htod(&v_host)?, self.n_cols, 1)?;

                if diff < tol { 
                    break; 
                }
            }

            // --- Eigenvalue Calculation ---
            // lambda = v^T * C * v
            let c_host = cov_mat.to_host()?;
            let mut lambda = 0.0;
            for i in 0..self.n_cols {
                let mut row_sum = 0.0;
                for j in 0..self.n_cols {
                    row_sum += c_host[i * self.n_cols + j] * v_host[j];
                }
                lambda += v_host[i] * row_sum;
            }

            // --- Store the Eigenvector ---
            // Place it into the appropriate column of our n_cols x k matrix
            for i in 0..self.n_cols {
                loadings_host[i * k + pc_idx] = v_host[i];
            }

            // --- Hotelling's Deflation ---
            // Update the covariance matrix in-place for the next iteration
            if pc_idx < k - 1 {
                cov_mat.deflate_mut(&v_mat, lambda)?;
            }
        }

        // 4. Push the final loadings matrix to the GPU
        let loadings_dev = stream.clone_htod(&loadings_host)?;
        let loadings_mat = Matrix::new(loadings_dev, self.n_cols, k)?;

        Ok(loadings_mat) 
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

        // 1. Create a 2x2 pseudo-covariance matrix
        // C = [2.0, 0.0]
        //     [0.0, 1.0]
        let c_host = vec![2.0, 0.0, 0.0, 1.0];
        let c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;
        let mut c_matrix = Matrix::new(c_dev, 2, 2)?;

        // 2. Create an eigenvector representing the first principal axis
        // v = [1.0]
        //     [0.0]
        let v_host = vec![1.0, 0.0];
        let v_dev: CudaSlice<f32> = stream.clone_htod(&v_host)?;
        let v_matrix = Matrix::new(v_dev, 2, 1)?;

        // 3. Deflate the matrix: C = C - lambda * v * v^T
        // Using lambda = 2.0 (the variance of the first axis)
        c_matrix.deflate_mut(&v_matrix, 2.0)?;

        // 4. Fetch the result back to host
        let mut result_host = vec![0.0f32; 4];
        stream.memcpy_dtoh(&c_matrix.data, &mut result_host)?;

        println!("After deflation: {:?}", result_host);

        // 5. Expected result: the variance on the first axis is removed
        // Expected: [0.0, 0.0, 0.0, 1.0]
        assert!((result_host[0] - 0.0).abs() < 1e-5);
        assert!((result_host[1] - 0.0).abs() < 1e-5);
        assert!((result_host[2] - 0.0).abs() < 1e-5);
        assert!((result_host[3] - 1.0).abs() < 1e-5);






        // 1. Create a 3x2 dataset (3 observations, 2 features)
        // This data is perfectly correlated linearly (y = x)
        // [1.0, 1.0]
        // [2.0, 2.0]
        // [3.0, 3.0]
        let n: usize = 10;
        let p: usize = 7;
        let mut x_host: Vec<f32> = (0..(n * p)).map(|x| x as f32).collect();
        let x_dev: CudaSlice<f32> = stream.clone_htod(&x_host)?;
        let x_matrix = Matrix::new(x_dev, n, p)?;

        // 2. Calculate the first 2 Principal Components
        // Using max_iter = 100 and tol = 1e-5
        let pcs_matrix = x_matrix.principal_components(5, 100, 1e-5)?;

        println!("x_matrix: {}", x_matrix);
        println!("pcs_matrix: {}", pcs_matrix);


        // 3. Fetch the loadings matrix back to the host
        // Result is a 2x2 matrix (n_cols x num_pcs)
        let result_host = pcs_matrix.to_host()?;

        // 4. Validate mathematically
        // For perfectly correlated data, PC1 lies perfectly on the diagonal.
        // Therefore, the normalized loadings for PC1 should be [1/sqrt(2), 1/sqrt(2)]
        let pc1_expected = std::f32::consts::FRAC_1_SQRT_2; // ~0.70710677

        // Note: Eigenvectors are invariant to sign (can point forwards or backwards),
        // so we test the absolute values to prevent flaky test failures.
        // PC1 is column 0 (indices 0 and 2 in row-major layout).
        assert!((result_host[0].abs() - pc1_expected).abs() < 1e-4);
        assert!((result_host[2].abs() - pc1_expected).abs() < 1e-4);

        // PC2 is column 1 (indices 1 and 3 in row-major layout).
        // Since it is orthogonal to PC1, its components should have opposite signs 
        // to equal zero dot product, but identical absolute magnitudes.
        assert!((result_host[1].abs() - pc1_expected).abs() < 1e-4);
        assert!((result_host[3].abs() - pc1_expected).abs() < 1e-4);

        Ok(())
    }
}