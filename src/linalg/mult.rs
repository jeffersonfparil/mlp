use crate::linalg::matrix::{Matrix, MatrixError};

impl Matrix {
    pub fn mult(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Matrix multiplication: {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows, other.cols)?;
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut x: f32 = 0.0;
                for k in 0..self.cols {
                    x += self.data[i][k] * other.data[k][j];
                }
                out.data[i][j] = x;
            }
        }
        Ok(out)
    }
    pub fn mult_scalar(&self, scalar: f32) -> Result<Matrix, MatrixError> {
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i][j] = self.data[i][j] * scalar;
            }
        }
        Ok(out)
    }
    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if (self.rows != other.rows) || (self.cols != other.cols) {
            return Err(MatrixError::DimensionMismatch(format!(
                "Hadamard product (element-wise product): {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        Ok(out)
    }
    pub fn div_scalar(&self, scalar: f32) -> Result<Matrix, MatrixError> {
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i][j] = self.data[i][j] / scalar;
            }
        }
        Ok(out)
    }
    pub fn hadamard_div(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if (self.rows != other.rows) || (self.cols != other.cols) {
            return Err(MatrixError::DimensionMismatch(format!(
                "Element-wise division: {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::zeros(self.rows, self.cols)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[i][j] = self.data[i][j] / other.data[i][j];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mult() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];
        let mat_a = Matrix::new(a)?;
        let mat_b = Matrix::new(b)?;
        let mat_mult = mat_a.mult(&mat_b)?;
        assert_eq!(mat_mult.data, vec![vec![58.0, 64.0], vec![139.0, 154.0],]);
        let mat_mult_scalar = mat_a.mult_scalar(-2.0)?;
        assert_eq!(
            mat_mult_scalar.data,
            vec![vec![-2.0, -4.0, -6.0], vec![-8.0, -10.0, -12.0]]
        );
        let mat_hadamard = mat_a.hadamard(&mat_a)?;
        let mat_hadamard_div = mat_a.hadamard_div(&mat_a)?;
        assert_eq!(
            mat_hadamard.data,
            vec![
                vec![1.0f32.powf(2.0), 2.0f32.powf(2.0), 3.0f32.powf(2.0)],
                vec![4.0f32.powf(2.0), 5.0f32.powf(2.0), 6.0f32.powf(2.0)]
            ]
        );
        assert_eq!(mat_hadamard_div.data, vec![vec![1.0; 3]; 2]);
        Ok(())
    }
}
