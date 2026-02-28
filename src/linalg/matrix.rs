use std::fmt;
use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<Vec<f32>>, // row-major order, i.e. data[row][col]
    pub rows: usize,
    pub cols: usize,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "Dimension: rows={}; cols={}\n⎡ {:.5} ... {:.5} ⎤\n⎢ ... ... ... ⎥\n⎣ {:.5} ... {:.5} ⎦\n",
            self.rows, self.cols, 
            self.data[0][0], self.data[0][self.cols - 1],
            self.data[self.rows - 1][0], self.data[self.rows - 1][self.cols - 1]
        )
    }
}

#[derive(Debug, PartialEq)]
pub enum MatrixError {
    DimensionMismatch(String),
    UnknownMatrixError(String),
}

/// Implement Error for MatrixError
impl Error for MatrixError {}

/// Implement std::fmt::Display for MatrixError
impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch(msg) => write!(f, "Matrix dimension mismatch: {}", msg),
            MatrixError::UnknownMatrixError(msg) => write!(f, "Unknown matrix error: {}", msg),
        }
    }
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![f32::NAN; cols]; rows],
            rows,
            cols,
        }
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    pub fn ones(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![1.0; cols]; rows],
            rows,
            cols,
        }
    }
    pub fn random(rows: usize, cols: usize, seed: u64) -> Result<Self, Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        let normal = Normal::new(0.0, 1.0)?;
        let x = (&mut rng).sample_iter(normal).take(rows * cols).collect::<Vec<f32>>();
        let mut out: Self = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                out.data[i][j] = x[i * cols + j];
            }
        }
        Ok(out)
    }
    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![0.0; size]; size];
        for i in 0..size {
            data[i][i] = 1.0;
        }
        Matrix {
            data,
            rows: size,
            cols: size,
        }
    }
    pub fn transpose(&self) -> Self {
        let mut data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }
        Matrix {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
    pub fn slice(&self, idx_rows: Vec<usize>, idx_cols: Vec<usize>) -> Self {
        let data = idx_rows
            .iter()
            .map(|&i| {
                idx_cols
                    .iter()
                    .map(|&j| self.data[i][j])
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        Matrix {
            data,
            rows: idx_rows.len(),
            cols: idx_cols.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix() {
        // TODO
        assert_eq!(true, true);
    }
}