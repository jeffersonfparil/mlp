use crate::linalg::matrix::{Matrix, MatrixError};

impl Matrix {
    pub fn add(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return MatrixError::DimensionMismatch(format!("Matrix-matrix addition: {}x{} and {}x{}", self.rows, self.cols, other.rows, other.cols));
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Ok(result)
    }
    pub fn add_scalar(&self, scalar: f32) -> Result<Matrix, MatrixError> {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + scalar;
            }
        }
        Ok(result)
    }
    pub fn add_row_vector(&self, vector: &Matrix) -> Result<Matrix, MatrixError> {
        if vector.rows != 1 || vector.cols != self.cols {
            return MatrixError::DimensionMismatch(format!("Matrix-row vector addition: {}x{} and {}x{}", self.rows, self.cols, vector.rows, vector.cols));
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + vector.data[0][j];
            }
        }
        Ok(result)
    }
    pub fn add_column_vector(&self, vector: &Matrix) -> Result<Matrix, MatrixError> {
        if vector.rows != self.rows || vector.cols != 1 {
            return MatrixError::DimensionMismatch(format!("Matrix-column vector addition: {}x{} and {}x{}", self.rows, self.cols, vector.rows, vector.cols));
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + vector.data[i][0];
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        // TODO
        assert_eq!(true, true);
    }
}