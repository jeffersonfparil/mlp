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
        let k = num_pcs.min(self.n_cols);
        // Centre
        let col_means_neg = self.colsummat()?.scalarmatmul(-(self.n_rows as f32))?;
        println!("col_means_neg: {}", col_means_neg);
        let centered_mat = self.colmatadd(&col_means_neg)?;
        println!("centered_mat: {}", centered_mat);
        // Covariance Matrix (X^T * X)
        let mut cov_mat = centered_mat.matmult0(&centered_mat)?;
        // Allocate host memory to store the final combined loadings matrix (row-major)
        let mut loadings_host = vec![0.0f32; self.n_cols * k];
        // Extract PCs sequentially
        for pc_idx in 0..k {
            let v_mat = {
                let mut v_mat = Matrix::new(stream.clone_htod(&vec![1.0f32; self.n_cols])?, self.n_cols, 1)?;
                for _ in 0..max_iter {
                    // v = C * v
                    // n = sqrt(sum(v^2))
                    // v = v / n
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
            // Eigenvalue Calculation
            // lambda = v^T * C * v
            let lambda = v_mat
                .matmult0(&cov_mat)?
                .matmul(&v_mat)?
                .to_host()?[0];
            // Store the Eigenvector
            // Place it into the appropriate column of our n_cols x k matrix
            let v_host = v_mat.to_host()?;
            for i in 0..self.n_cols {
                loadings_host[i * k + pc_idx] = v_host[i];
            }
            // Hotelling's Deflation (C = C - lambda * v * v^T)
            // Update the covariance matrix in-place for the next iteration
            if pc_idx < k - 1 {
                // cov_mat.deflate_mut(&v_mat, lambda)?;
                cov_mat = cov_mat.elementwisematadd(
                    &v_mat
                        .matmul0t(&v_mat)?
                        .scalarmatmul(-lambda)?
                )?;
            }
        }

        // Push the final loadings matrix to the GPU
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
        let mut x_host: Vec<f32> = (0..(n * p)).map(|x| 1.1_f32.powf(x as f32)).collect();
        let x_dev: CudaSlice<f32> = stream.clone_htod(&x_host)?;
        let x_matrix = Matrix::new(x_dev, n, p)?;

        // 2. Calculate the first 2 Principal Components
        // Using max_iter = 100 and tol = 1e-5
        let pcs_matrix = x_matrix.principal_components(1, 100, 1e-5)?;

        println!("x_matrix: {}", x_matrix);
        println!("pcs_matrix: {}", pcs_matrix);


        // 3. Fetch the loadings matrix back to the host
        // Result is a 2x2 matrix (n_cols x num_pcs)
        let result_pc1: Vec<f32> = pcs_matrix.to_host()?;
        let expected_pc1: Vec<f32> = vec![0.2739837, 0.3013821, 0.3315203, 0.3646723, 0.4011395, 0.4412535, 0.4853788];
        println!("result_pc1: {:?}", result_pc1);
        println!("expected_pc1: {:?}", expected_pc1);
        for i in 0..p {
            assert!((result_pc1[i] - expected_pc1[i]).abs() < 0.001);
        }


        Ok(())
    }
}