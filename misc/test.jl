using Random, CUDA, LinearAlgebra, Distributions, StatsBase, UnicodePlots, ProgressMeter
# # Activation function and its derivative
# relu(x) = max(x, 0.0)
# δrelu(x) = x > 0.0 ? 1.0 : 0.0
# linear(x) = x
# δlinear(x) = 1.0
# # Cost (loss) function and its gradient w.r.t. predicted output (mean over neurons)
# mse(N̂.y, y) = mean(0.5 .* (N̂.y .- y) .^ 2)
# δmse(N̂.y, y) = (N̂.y .- y) ./ length(y)   # dL/dŷ_i = (ŷ_i - y_i) / n

mutable struct Neuron
    y::CuArray{Float64,1}
    W::CuArray{Float64,2}
    a::CuArray{Float64,1}
    b::CuArray{Float64,1}
    F::Function
    δF::Function
    C::Function
    δC::Function
    function Neuron(;
        n::Int64=20,
        p::Int64=15,
        F::Union{Nothing,Function}=nothing,
        δF::Union{Nothing,Function}=nothing,
        C::Union{Nothing,Function}=nothing,
        δC::Union{Nothing,Function}=nothing,
        seed::Int64=42,
    )::Neuron
        # n::Int64 = 20; p::Int64 = 15; F::Union{Nothing, Function} = nothing; δF::Union{Nothing, Function} = nothing; C::Union{Nothing, Function} = nothing; δC::Union{Nothing, Function} = nothing; seed::Int64 = 42
        Random.seed!(seed)
        D = Normal(0.0, 1.0)
        W = CuArray(rand(D, n, p))
        b = CuArray(rand(D, n))
        a = CuArray(rand(D, p))
        F = isnothing(F) ? x -> x : F
        δF = isnothing(δF) ? x -> 1 : δF
        C = isnothing(C) ? (a, b) -> mean(0.5 * (a - b)^2) : C
        δC = isnothing(δC) ? (a, b) -> (a - b) / length(a) : δC
        y = F.((W * a) .+ b)
        new(y, W, a, b, F, δF, C, δC)
    end
end

###########################################################
###########################################################
###########################################################
# Training via gradient descent
function gd(;
    y::CuArray{Float64,1},
    a::CuArray{Float64,1},
    learning_rate::Float64=1.00e-2,
    n_epochs::Int64=10_000,
    F::Union{Nothing,Function}=nothing,
    δF::Union{Nothing,Function}=nothing,
    C::Union{Nothing,Function}=nothing,
    δC::Union{Nothing,Function}=nothing,
    seed::Int64=42,
    verbose::Bool=false,
)::Neuron
    # N = Neuron(seed=12345)
    # y = N.y
    # a = N.a
    # learning_rate = 1e-2
    # n_epochs = 10
    # F = nothing
    # δF = nothing
    # C = nothing
    # δC = nothing
    # seed = 42
    # verbose = true
    n = length(y)
    p = length(a)
    F = isnothing(F) ? x -> x : F
    δF = isnothing(δF) ? x -> 1 : δF
    C = isnothing(C) ? (a, b) -> mean(0.5 * (a - b)^2) : C
    δC = isnothing(δC) ? (a, b) -> (a - b) / length(a) : δC
    N̂ = begin
        Random.seed!(seed)
        seed = Int(ceil(100 * rand()))
        N̂ = Neuron(n=n, p=p, F=F, δF=δF, C=C, δC=δC, seed=seed)
        N̂.a = a
        N̂
    end
    t = collect(1:n_epochs)
    c = []
    if verbose
        pb = ProgressMeter.Progress(n_epochs, desc="Training via gradient descent")
    end
    for i in t
        # i = 1
        # forward
        # v̂: pre-activation vector (n,)
        # N̂.W: weights matrix (n x p)
        # a: input vector (p,)
        # N̂.b: bias vector (n,)
        v̂ = N̂.W * a .+ N̂.b # matrix-vector product -> n-vector
        # N̂.y: post-activation (predictions) (n,)
        # F: elementwise activation (e.g. relu, sigmoid)
        N̂.y = N̂.F.(v̂)
        # backprop (elementwise)
        # dL_dŷ: dL/dN̂.y, derivative of loss w.r.t. outputs (n,)
        #   δC should return an n-vector of ∂L/∂N̂.y_i for the current sample
        dL_dŷ = N̂.δC.(N̂.y, y) # n-vector
        # dŷ_dv̂: dN̂.y/dv̂, elementwise derivative of activation evaluated at v̂ (n,)
        #   Always compute the activation derivative at the pre-activation v̂ (not at N̂.y),
        #   unless you have an activation whose derivative is expressed via N̂.y (some libs do that).
        dŷ_dv̂ = N̂.δF.(v̂) # n-vector
        # Chain rule: dL/dv̂ = (dL/dN̂.y) .* (dN̂.y/dv̂) --> notice that this is element-wise
        dL_dv̂ = dL_dŷ .* dŷ_dv̂ # n-vector = ∂L/∂v̂ (often called "delta" per neuron)
        # gradients: outer product for weights, bias gradient is the delta
        # ∇N̂.W: n x p matrix. For a single sample, it's (n x 1) * (1 x p) -> n x p
        # Using outer product: each neuron's delta times each input feature.
        ∇W = dL_dv̂ * a' # n x p
        # ∇N̂.b: same shape as bias (n,). For a single sample it's just the delta.
        # For a mini-batch, you'd sum (or average) the deltas across examples.
        ∇b = dL_dv̂ # n-vector
        # gradient descent update
        # learning_rate is a scalar. Using elementwise .-= is fine; learning_rate * ∇N̂.W also works without dot.
        N̂.W .-= learning_rate .* ∇W
        N̂.b .-= learning_rate .* ∇b
        # report loss (move to CPU for printing)
        @show push!(c, sum(C.(N̂.y, y)))
        # println("i=$i; c=$c")
        # UnicodePlots.scatterplot(t[1:i], c)
        # display(UnicodePlots.scatterplot(t[1:i], c[1:i]))
        if verbose
            ProgressMeter.next!(pb)
        end
    end
    if verbose
        ProgressMeter.finish!(pb)
    end
    N̂.y = F.(N̂.W * N̂.a .+ N̂.b)
    N̂
end

N = Neuron(seed=Int64(ceil(100 * rand())))
# N̂ = Neuron(seed=Int64(ceil(100 * rand())))
N̂ = gd(y=N.y, a=N.a, n_epochs=10, seed=Int64(ceil(100 * rand())), verbose=true)

sum(C.(N̂.y, N.y))
cor(Array(N̂.y), Array(N.y))
UnicodePlots.scatterplot(Array(N̂.y), Array(N.y))

if false
    Ŵ_vec = reshape(Array(N̂.W), n * p, 1)[:, 1]
    W_vec = reshape(Array(N.W), n * p, 1)[:, 1]
    b̂_vec = Array(N̂.b)
    b_vec = Array(N.b)
    cor(W_vec, Ŵ_vec)
    cor(b_vec, b̂_vec)
    UnicodePlots.scatterplot(W_vec, Ŵ_vec)
    UnicodePlots.scatterplot(b_vec, b̂_vec)
end

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
C(N̂.y, y) = mean(0.5 .* (N̂.y .- y) .^ 2)
δC(N̂.y, y) = (N̂.y .- y) ./ length(y)   # dL/dŷ_i = (ŷ_i - y_i) / n
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
    N̂.W = CuArray(rand(MvNormal(α["W_μ"], α["W_Σ"]), n)')
    N̂.b = CuArray(rand(Normal(α["b_μ"], α["b_σ"]), n))
    Dict("N̂.W" => N̂.W, "N̂.b" => N̂.b)
end

function posterior_likelihood_ish(; N̂.W, N̂.b, y)
    N̂.y = F.(N̂.W * a .+ N̂.b)
    C(N̂.y, y)
end

proposal_scale = 0.01
accepted = 0

priors = sample_priors(α)
N̂.W = priors["N̂.W"]
N̂.b = priors["N̂.b"]
L = posterior_likelihood_ish(N̂.W=N̂.W, N̂.b=N̂.b, y=y)

N̂.W_trace = CuArray{Float64}[]
N̂.b_trace = CuArray{Float64}[]

n_iterations = 1_000_000
n_burins = 100_000

pb = ProgressMeter.Progress(n_iterations)
for t in 1:n_iterations
    proposal_N̂.W = N̂.W .+ (proposal_scale .* CuArray(randn(n, p)))
    proposal_N̂.b = N̂.b .+ (proposal_scale .* CuArray(randn(n)))
    proposal_L = posterior_likelihood_ish(N̂.W=proposal_N̂.W, N̂.b=proposal_N̂.b, y=y)

    if (proposal_L - L) < rand(Uniform(0.0, 1.0))
        N̂.W = proposal_N̂.W
        N̂.b = proposal_N̂.b
        L = proposal_L
        accepted += 1
        if t > n_burins
            push!(N̂.W_trace, N̂.W)
            push!(N̂.b_trace, N̂.b)
        end
    end
    ProgressMeter.next!(pb)
end
ProgressMeter.finish!(pb)
@show accepted

N̂.W = mean(N̂.W_trace)
N̂.b = mean(N̂.b_trace)

N̂.y = F.(N̂.W * a .+ N̂.b)
C(y, N̂.y)
hcat(y, N̂.y)
cor(Array(y), Array(N̂.y))
UnicodePlots.scatterplot(Array(y), Array(N̂.y))

if false
    N̂.W_vec = reshape(Array(N̂.W), n * p, 1)[:, 1]
    W_vec = reshape(Array(W), n * p, 1)[:, 1]
    N̂.b_vec = Array(N̂.b)
    b_vec = Array(b)
    cor(N̂.W_vec, W_vec)
    cor(N̂.b_vec, b_vec)
    UnicodePlots.scatterplot(N̂.W_vec, W_vec)
    UnicodePlots.scatterplot(N̂.b_vec, b_vec)
end


