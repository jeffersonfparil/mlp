use crate::linalg::matrix::{Matrix, MatrixError};

impl Matrix {
    pub fn mult(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Matrix multiplication: {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out = Matrix::new(self.rows, self.cols);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mult() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]];
        let mat_a = Matrix {
            data: a,
            rows: 2,
            cols: 3,
        };
        let mat_b = Matrix {
            data: b,
            rows: 2,
            cols: 3,
        };
        Ok(())
    }
}
