using CUDA, LinearAlgebra, Distributions, StatsBase, UnicodePlots

n = 20 # output nodes for this neuron
p = 15 # input nodes for this neuron
D = Normal(0.0, 1.0)

# parameters and input (on GPU)
W = CuArray(rand(D, n, p))
a = CuArray(rand(D, p))       # input vector (p)
b = CuArray(rand(D, n))       # bias vector (n)

# activation (ReLU) and its derivative (evaluate at pre-activation)
F(x) = max(x, 0.0)
δF(x) = x > 0.0 ? 1.0 : 0.0

# F(x) = x
# δF(x) = 1.0


# forward pass (ground truth)
A = W * a .+ b                # n-vector
y = F.(A)

# quick histograms (copy to CPU)
UnicodePlots.histogram(Array(A))
UnicodePlots.histogram(Array(y))

# loss and its gradient w.r.t. predicted output (mean over neurons)
C(ŷ, y) = mean(0.5 .* (ŷ .- y) .^ 2)
δC(ŷ, y) = (ŷ .- y) ./ length(y)   # dL/dŷ_i = (ŷ_i - y_i) / n

# initialize model parameters to fit
Ŵ = CuArray(rand(D, n, p))
b̂ = CuArray(rand(D, n))

lr = 1e-2   # learning rate
n_epochs = 10_000
t = collect(1:n_epochs)
c = []
@time for i in t
    # i = 100
    # forward
    # Â: pre-activation vector (n,)
    # Ŵ: weights matrix (n x p)
    # a: input vector (p,)
    # b̂: bias vector (n,)
    Â = Ŵ * a .+ b̂ # matrix-vector product -> n-vector
    # ŷ: post-activation (predictions) (n,)
    # F: elementwise activation (e.g. relu, sigmoid)
    ŷ = F.(Â)
    # backprop (elementwise)
    # dL_dŷ: dL/dŷ, derivative of loss w.r.t. outputs (n,)
    #   δC should return an n-vector of ∂L/∂ŷ_i for the current sample
    dL_dŷ = δC(ŷ, y) # n-vector
    # dŷ_dÂ: dŷ/dÂ, elementwise derivative of activation evaluated at Â (n,)
    #   Always compute the activation derivative at the pre-activation Â (not at ŷ),
    #   unless you have an activation whose derivative is expressed via ŷ (some libs do that).
    dŷ_dÂ = δF.(Â) # n-vector
    # Chain rule: dL/dÂ = (dL/dŷ) .* (dŷ/dÂ) --> notice that this is element-wise
    dL_dÂ = dL_dŷ .* dŷ_dÂ # n-vector = ∂L/∂Â (often called "delta" per neuron)
    # gradients: outer product for weights, bias gradient is the delta
    # ∇Ŵ: n x p matrix. For a single sample, it's (n x 1) * (1 x p) -> n x p
    # Using outer product: each neuron's delta times each input feature.
    ∇Ŵ = dL_dÂ * a' # n x p
    # ∇b̂: same shape as bias (n,). For a single sample it's just the delta.
    # For a mini-batch, you'd sum (or average) the deltas across examples.
    ∇b̂ = dL_dÂ # n-vector
    # gradient descent update
    # lr is a scalar. Using elementwise .-= is fine; lr * ∇Ŵ also works without dot.
    Ŵ .-= lr .* ∇Ŵ
    b̂ .-= lr .* ∇b̂
    # report loss (move to CPU for printing)
    push!(c, C(ŷ, y))
    # println("i=$i; c=$c")
    # UnicodePlots.scatterplot(t[1:i], c)
    # display(UnicodePlots.scatterplot(t[1:i], c[1:i]))
end
# UnicodePlots.scatterplot(t, c)

ŷ = F.(Ŵ * a .+ b̂)
hcat(ŷ, y)
C(ŷ, y)
cor(Array(ŷ), Array(y))
UnicodePlots.scatterplot(Array(ŷ), Array(y))

Ŵ_vec = reshape(Array(Ŵ), n * p, 1)[:, 1]
W_vec = reshape(Array(W), n * p, 1)[:, 1]
b̂_vec = Array(b̂)
b_vec = Array(b)
cor(Ŵ_vec, W_vec)
cor(b̂_vec, b_vec)
UnicodePlots.scatterplot(Ŵ_vec, W_vec)
UnicodePlots.scatterplot(b̂_vec, b_vec)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Bayesian-ish optimisation
Ŵ = CuArray(rand(D, n, p))
b̂ = CuArray(rand(D, n))
ŷ = F.(Ŵ * a + b̂)

α = hcat(
    rand(Normal(0, 1), 4),
    rand(Uniform(0, 1), 4),
)
priors = begin
    M = rand(Normal(α[1, 1], α[1, 2]), n, p)
    U = begin
        u = rand(Normal(α[2, 1], α[2, 2]), n)
        U = u * u'
        U[diagind(U)] .+= 1.00
        U
    end
    V = begin
        v = rand(Normal(α[3, 1], α[3, 2]), p)
        V = v * v'
        V[diagind(V)] .+= 1.00
        V
    end
    Ŵ_prior = CuArray(rand(MatrixNormal(M, U, V)))
    b̂_prior = CuArray(rand(Normal(α[4, 1], α[4, 2]), n))
    (Ŵ_prior, b̂_prior)
end
ŷ = F.(priors[1] * a + priors[2])
likelihood = C(ŷ, y)
posterior = likelihood .* priors
hcat(reshape(priors[1], n * p, 1), reshape(posterior[1], n * p, 1))
hcat(priors[2], posterior[2])

# TODO: continue working on the iterative Bayesian updates above
