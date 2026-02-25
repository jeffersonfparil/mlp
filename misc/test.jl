using CUDA, LinearAlgebra, Distributions, StatsBase, UnicodePlots, ProgressMeter
# Simulate
n = 20 # output nodes for this neuron
p = 15 # input nodes for this neuron
D = Normal(0.0, 1.0)
W = CuArray(rand(D, n, p))
a = CuArray(rand(D, p))
b = CuArray(rand(D, n))
v = W * a .+ b
y = F.(v)
# UnicodePlots.histogram(Array(v))
# UnicodePlots.histogram(Array(y))
# Activation function and its derivative
# F(x) = max(x, 0.0)
# δF(x) = x > 0.0 ? 1.0 : 0.0
F(x) = x
δF(x) = 1.0
# Cost (loss) function and its gradient w.r.t. predicted output (mean over neurons)
C(ŷ, y) = mean(0.5 .* (ŷ .- y) .^ 2)
δC(ŷ, y) = (ŷ .- y) ./ length(y)   # dL/dŷ_i = (ŷ_i - y_i) / n

###########################################################
###########################################################
###########################################################
# Training via gradient descent
Ŵ = CuArray(rand(D, n, p))
b̂ = CuArray(rand(D, n))
lr = 1e-2
n_epochs = 10_000
t = collect(1:n_epochs)
c = []
@time for i in t
    # i = 100
    # forward
    # v̂: pre-activation vector (n,)
    # Ŵ: weights matrix (n x p)
    # a: input vector (p,)
    # b̂: bias vector (n,)
    v̂ = Ŵ * a .+ b̂ # matrix-vector product -> n-vector
    # ŷ: post-activation (predictions) (n,)
    # F: elementwise activation (e.g. relu, sigmoid)
    ŷ = F.(v̂)
    # backprop (elementwise)
    # dL_dŷ: dL/dŷ, derivative of loss w.r.t. outputs (n,)
    #   δC should return an n-vector of ∂L/∂ŷ_i for the current sample
    dL_dŷ = δC(ŷ, y) # n-vector
    # dŷ_dv̂: dŷ/dv̂, elementwise derivative of activation evaluated at v̂ (n,)
    #   Always compute the activation derivative at the pre-activation v̂ (not at ŷ),
    #   unless you have an activation whose derivative is expressed via ŷ (some libs do that).
    dŷ_dv̂ = δF.(v̂) # n-vector
    # Chain rule: dL/dv̂ = (dL/dŷ) .* (dŷ/dv̂) --> notice that this is element-wise
    dL_dv̂ = dL_dŷ .* dŷ_dv̂ # n-vector = ∂L/∂v̂ (often called "delta" per neuron)
    # gradients: outer product for weights, bias gradient is the delta
    # ∇Ŵ: n x p matrix. For a single sample, it's (n x 1) * (1 x p) -> n x p
    # Using outer product: each neuron's delta times each input feature.
    ∇Ŵ = dL_dv̂ * a' # n x p
    # ∇b̂: same shape as bias (n,). For a single sample it's just the delta.
    # For a mini-batch, you'd sum (or average) the deltas across examples.
    ∇b̂ = dL_dv̂ # n-vector
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

# Ŵ_vec = reshape(Array(Ŵ), n * p, 1)[:, 1]
# W_vec = reshape(Array(W), n * p, 1)[:, 1]
# b̂_vec = Array(b̂)
# b_vec = Array(b)
# cor(Ŵ_vec, W_vec)
# cor(b̂_vec, b_vec)
# UnicodePlots.scatterplot(Ŵ_vec, W_vec)
# UnicodePlots.scatterplot(b̂_vec, b_vec)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Bayesian-ish optimisation

using CUDA, LinearAlgebra, Distributions, StatsBase, UnicodePlots, ProgressMeter
# Simulate
n = 20 # output nodes for this neuron
p = 15 # input nodes for this neuron
D = Normal(0.0, 1.0)
W = CuArray(rand(D, n, p))
a = CuArray(rand(D, p))
b = CuArray(rand(D, n))
# Activation function and its derivative
F(x) = max(x, 0.0)
δF(x) = x > 0.0 ? 1.0 : 0.0
# F(x) = x
# δF(x) = 1.0
# Cost (loss) function and its gradient w.r.t. predicted output (mean over neurons)
C(ŷ, y) = mean(0.5 .* (ŷ .- y) .^ 2)
δC(ŷ, y) = (ŷ .- y) ./ length(y)   # dL/dŷ_i = (ŷ_i - y_i) / n
# Simulate output
v = W * a .+ b
y = F.(v)
# UnicodePlots.histogram(Array(v))
# UnicodePlots.histogram(Array(y))

# Hyperparameters for the priors
α = Dict(
    "W_μ" => rand(Normal(0.0, 1.0), p),
    "W_Σ" => begin
        W_Σ = rand(Wishart(p - 1,
            begin
                v = rand(Beta(2.0, 2.0), p)
                V = v * v'
                V[diagind(V)] .+= maximum(abs.(v))
                V
            end
        ))
        W_Σ[diagind(W_Σ)] .+= maximum(abs.(W_Σ))
        W_Σ
    end,
    "b_μ" => 0.0,
    "b_σ" => 1.0,
)

# Sample the priors using the hyperparameters above. For simplicity, we use Normal priors for all parameters, 
# but you can choose different distributions as needed.
# The key is to ensure that the priors are appropriately defined for the parameters they represent (e.g. positive variance for covariance matrices).
function sample_priors(α)
    Ŵ = CuArray(rand(MvNormal(α["W_μ"], α["W_Σ"]), n)')
    b̂ = CuArray(rand(Normal(α["b_μ"], α["b_σ"]), n))
    Dict("Ŵ" => Ŵ, "b̂" => b̂)
end

function posterior_likelihood_ish(; Ŵ, b̂, y)
    ŷ = F.(Ŵ * a .+ b̂)
    C(ŷ, y)
end

proposal_scale = 0.01
accepted = 0

priors = sample_priors(α)
Ŵ = priors["Ŵ"]
b̂ = priors["b̂"]
L = posterior_likelihood_ish(Ŵ=Ŵ, b̂=b̂, y=y)

Ŵ_trace = CuArray{Float64}[]
b̂_trace = CuArray{Float64}[]

n_iterations = 1_000_000
n_burins = 100_000

pb = ProgressMeter.Progress(n_iterations)
for t in 1:n_iterations
    proposal_Ŵ = Ŵ .+ (proposal_scale .* CuArray(randn(n, p)))
    proposal_b̂ = b̂ .+ (proposal_scale .* CuArray(randn(n)))
    proposal_L = posterior_likelihood_ish(Ŵ=proposal_Ŵ, b̂=proposal_b̂, y=y)

    if (proposal_L - L) < rand(Uniform(0.0, 1.0))
        Ŵ = proposal_Ŵ
        b̂ = proposal_b̂
        L = proposal_L
        accepted += 1
        if t > n_burins
            push!(Ŵ_trace, Ŵ)
            push!(b̂_trace, b̂)
        end
    end
    ProgressMeter.next!(pb)
end
ProgressMeter.finish!(pb)
@show accepted

Ŵ = mean(Ŵ_trace)
b̂ = mean(b̂_trace)

ŷ = F.(Ŵ * a .+ b̂)
C(y, ŷ)
hcat(y, ŷ)
cor(Array(y), Array(ŷ))
UnicodePlots.scatterplot(Array(y), Array(ŷ))




