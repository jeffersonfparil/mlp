using CUDA, LinearAlgebra, Distributions, StatsBase

# Defining a single neuron:
#   - input: a
#   - output: y
#   - weights: W
#   - biases: b
#   - activation function: F
n = 10_000
p = 1_000
D = Normal(0.0, 1.0)
W = CuArray(rand(D, n, p))
a = CuArray(rand(D, p))
b = CuArray(rand(D, n))
function F(x)
    x -> x > 0.0 ? x : 0.0
end
y = F.((W * a) + b)

# Adjusting the weights and biases using a cost function
function C(ŷ, y)

end
