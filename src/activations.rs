use crate::linalg::matrix::Matrix;
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Sigmoid,
    HyperbolicTangent,
    ReLU,
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Activation::Sigmoid => {
                write!(f, "Sigmoid")
            }
            Activation::HyperbolicTangent => {
                write!(f, "HyperbolicTangent")
            }
            Activation::ReLU => {
                write!(f, "ReLU")
            }
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ActivationError {
    UnimplementedActivation,
    UnimplementedActivationDerivative,
}

/// Implement Error for ActivationError
impl Error for ActivationError {}

/// Implement std::fmt::Display for ActivationError
impl fmt::Display for ActivationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActivationError::UnimplementedActivation => {
                write!(f, "Activation error unknown activation function")
            }
            ActivationError::UnimplementedActivationDerivative => {
                write!(f, "Activation error unknown activation function")
            }
        }
    }
}

pub fn sigmoid(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}
pub fn sigmoidderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}
pub fn hyperbolictangent(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}
pub fn hyperbolictangentderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}
pub fn relu(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}
pub fn reluderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    unimplemented!()
}

impl Activation {
    pub fn activate(&self, a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Activation::Sigmoid => sigmoid(a),
            Activation::HyperbolicTangent => hyperbolictangent(a),
            Activation::ReLU => relu(a),
            // _ => {
            //     return Err(Box::new(ActivationError::UnimplementedActivation));
            // }
        }
    }
    pub fn derivative(&self, a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Activation::Sigmoid => sigmoidderivative(a),
            Activation::HyperbolicTangent => hyperbolictangentderivative(a),
            Activation::ReLU => reluderivative(a),
            // _ => {
            //     return Err(Box::new(ActivationError::UnimplementedActivationDerivative));
            // }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_sigmoid() {
        // TODO
        assert_eq!(true, true);
    }
}
