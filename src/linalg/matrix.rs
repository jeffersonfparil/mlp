use std::fmt;
use std::error::Error;

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
    DimensionMismatch,
    UnimplementedMatrix,
}

/// Implement Error for MatrixError
impl Error for MatrixError {}

/// Implement std::fmt::Display for MatrixError
impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch => {
                write!(f, "Matrix dimension mismatch")
            }
            MatrixError::UnimplementedMatrix => {
                write!(f, "Unknown matrix function")
            }
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