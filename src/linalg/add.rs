use crate::linalg::matrix::{Matrix, MatrixError};

impl Matrix {
    pub fn add(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Matrix-matrix addition: {}x{} and {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
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
            return Err(MatrixError::DimensionMismatch(format!(
                "Matrix-row vector addition: {}x{} and {}x{}",
                self.rows, self.cols, vector.rows, vector.cols
            )));
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
            return Err(MatrixError::DimensionMismatch(format!(
                "Matrix-column vector addition: {}x{} and {}x{}",
                self.rows, self.cols, vector.rows, vector.cols
            )));
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + vector.data[i][0];
            }
        }
        Ok(result)
    }
    pub fn sum(&self, skip_nan: bool) -> Result<f32, MatrixError> {
        let mut s: f32 = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                let x = self.data[i][j];
                if skip_nan & x.is_nan() {
                    continue;
                }
                s += x;
            }
        }
        Ok(s)
    }
    pub fn sum_rows(&self, skip_nan: bool) -> Result<Vec<f32>, MatrixError> {
        let mut s: Vec<f32> = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let x = self.data[i][j];
                if skip_nan & x.is_nan() {
                    continue;
                }
                s[i] += x;
            }
        }
        Ok(s)
    }
    pub fn sum_cols(&self, skip_nan: bool) -> Result<Vec<f32>, MatrixError> {
        let mut s: Vec<f32> = vec![0.0; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let x = self.data[i][j];
                if skip_nan & x.is_nan() {
                    continue;
                }
                s[j] += x;
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() -> Result<(), Box<dyn std::error::Error>> {
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
        let mat_add = mat_a.add(&mat_b)?;
        assert_eq!(
            mat_add.data,
            vec![vec![8.0, 10.0, 12.0], vec![14.0, 16.0, 18.0]]
        );
        let mat_add_scalar = mat_a.add_scalar(10.0)?;
        assert_eq!(
            mat_add_scalar.data,
            vec![vec![11.0, 12.0, 13.0], vec![14.0, 15.0, 16.0]]
        );
        let row_vector = mat_a.slice(vec![0], vec![0, 1, 2]);
        let mat_add_row_vector = mat_a.add_row_vector(&row_vector)?;
        assert_eq!(
            mat_add_row_vector.data,
            vec![vec![2.0, 4.0, 6.0], vec![5.0, 7.0, 9.0]]
        );
        let column_vector = mat_a.slice(vec![0, 1], vec![0]);
        let mat_add_column_vector = mat_a.add_column_vector(&column_vector)?;
        assert_eq!(
            mat_add_column_vector.data,
            vec![vec![2.0, 3.0, 4.0], vec![8.0, 9.0, 10.0]]
        );
        let total = mat_a.sum(false)?;
        assert_eq!(total, 21.0);
        let total_per_row = mat_a.sum_rows(false)?;
        assert_eq!(total_per_row, vec![6.0f32, 15.0f32]);
        let total_per_col = mat_a.sum_cols(false)?;
        assert_eq!(total_per_col, vec![5.0f32, 7.0f32, 9.0f32]);
        Ok(())
    }
}
