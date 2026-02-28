use crate::linalg::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct Neuron {
    outputs: Matrix,
    weights: Matrix,
    inputs: Matrix,
    biases: Matrix,
}