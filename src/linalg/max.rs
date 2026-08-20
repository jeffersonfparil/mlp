use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const ELEMENTWISEMATMAX: &str = "
    extern \"C\" __global__ void cuElementwiseMatMax(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Matrix elmentwise maximum between 2 matrices kernel implementation
        // Arguments:
        //  - A: input matrix 1 (n_rows x n_cols)
        //  - B: input matrix 2 (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in all 3 matrices
        //  - n_cols: number of columns in all 3 matrices
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            C[idx] = fmaxf(A[idx], B[idx]);
        }
    }
";

impl Matrix {
    pub fn elementwisematmax(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_rows != b.n_rows) | (self.n_cols != b.n_cols) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `elementwisematmax`: self.n_rows ({}) != b.n_rows ({}) and/or self.n_cols ({}) != b.n_cols ({})",
                self.n_rows, b.n_rows, self.n_cols, b.n_cols
            ))));
        }
        let f: CudaFunction = self.get_cached_kernel("cuElementwiseMatMax", ELEMENTWISEMATMAX)?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&b.data);
        builder.arg(&mut out_dev);
        builder.arg(&n_rows);
        builder.arg(&n_cols);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }
}