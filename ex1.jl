using StatsBase
using Random
using Distributions
using CairoMakie
Random.seed!(0) # fix the seed to have consistent simulations

# P(X ∈ μ ± 2σ) ≈ 95% 
μtrue = [0.0, 4.0, 8.0]


###### 1) Simultate formward model 1\3 ∑_i N(μ[i],1)
Nobs = 1000 # number observations
function samplemixtue(N, μ)
    obs = zeros(N) # initialise
    kk = zeros(Int64, N) # initialise    
    for i ∈ 1:N
        k = rand(1:length(μtrue)) 
        kk[i] = k
        obs[i] = randn() + μ[k]
    end
    return obs, kk
end
obs, kk = samplemixtue(Nobs, μtrue)

struct Prior
    μ0::Float64
    σ20::Float64
    π0::Vector{Float64}
end


struct Entropy
    M::Matrix{Float64} # allocation matrix (N x K)
    p::Vector{Float64} # entropy probabilities
end




function gibbs(x, z0, μ0, Niter, P0::Prior; version = :standard)
    if version == :standard 
        println("Standard gibbs") 
    elseif version == :adapt 
        println("Entropy based gibbs")
    else
        error("This version has not been implemented")
    end
    n = length(obs)
    _M = probs!(zeros(length(obs), length(μ0)), μ0) # initialise allocation matrix
    _p = [1/n for i ∈ axes(_M, 1)] # initialise probabilities 
    E = Entropy(_M, _p) 
    zz = fill(zeros(Int64, length(obs)), Niter) # initilise container for output
    mm = fill(zeros(Float64, length(μ0)), Niter) # initilise container for output
    zz[1] = z0
    mm[1] = μ0
    z = copy(z0)
    μ = copy(μ0)
    for i in 2:Niter
        z = gibbsZ!(P0, E, x, z, μ, i, version)
        μ = gibbsμ!(P0, x, z, μ)
        zz[i] = copy(z)
        mm[i] = copy(μ)
    end
    return zz, mm
end


function gibbsZ!(P::Prior, E::Entropy, x, z, μ, i, version)
    p = E.p
    if version == :adapt 
        (i%1000 ==0  && i > 500) == 0
        E = update_p!(E, i) # update probabilities to select each individual
    end
    j = sample(Weights(p))  # not efficient, see ?sample. For improving: see https://github.com/TuringLang/AdvancedPS.jl/blob/master/src/resampling.jl
    E = update_row!(E, j, x, μ)
    z[j] = sample(Weights(E.M[j,:].*P.π0))
    return z
end

# check carefully the adaptation below
function update_p!(E::Entropy, i)  
    M = E.M
    E.p .= E.p.*(i - 1)/i/100 .+ (1/i)/100*(0.95*[entropy(M[i,:]) for i ∈ axes(M, 1)] .+ 0.05*1/length(E.p))
    return E
end

# TODO: work with log probabilities.
function update_row!(E, j, x, μ) # update allocation matrix 
    p = [exp(-(x[j] - μi)^2/2) for μi ∈ μ]  # should not need to normalise
    E.M[j, :] .= p 
    return E
end


function gibbsμ!(P::Prior, x, z, μ) 
    for i ∈ eachindex(μ)
        ind = z .== i
        xind = x[ind]
        tot = sum(ind)
        prec = tot + 1/P.σ20
        m = (sum(xind) + P.μ0/P.σ20)/prec
        μ[i] = randn()/sqrt(prec) + m 
    end
    return μ
end

# true z prob under μtrue
function probs!(Pz, μ)
    for i in eachindex(obs)
        pi = [exp(-(obs[i] - μj)^2/2) for μj in μ]
        ptot = sum(pi)
        pi ./= ptot # normalise
        Pz[i, :] .= pi 
    end
    Pz
end


function entropy(p)
    -sum(pi*log(pi) for pi in p)
end

function bayesfactors!(Cj, j, x, μ)   ### CHECK SMC TRICKS, LOOK AT SequentialMonteCarlo.jl
    p = [exp(-(x[j] - μi)^2/2) for μi in μ] 
    # p = p./sum(p)
    Cj.p .= p 
    return Cj
end

initialization = :random
if initialization == :random
    println("random allocation of cluster and random initialization of means")
    z0 = [rand(1:length(μtrue)) for _ in eachindex(obs)]
    μ0 = randn(length(μtrue)) 
elseif initialization == :true
    println("initialization from the true parameters")
    z0 = kk
    μ0 = μtrue
else
    error("please specify initialization")
end
Niter = 10^5
version =  :adapt
# version = :standard
P0 = Prior(0.0, 100.0, [1/3, 1/3, 1/3])
@time trace = gibbs(obs, z0, μ0, Niter, P0, version = version)
zz, mm = trace


#### PLOTS AND DISPLAYES
display([[sum(getindex.(trace[2][end-1000:end], i))/1001 for i in 1:length(μtrue)]  μtrue ])
A = [[sum(getindex.(trace[2][end-1000:end], i))/1001 for i in 1:length(μtrue)]  μtrue] 
println("check accuracy: $(sum(A[:,1]) - sum(A[:,2]))")

# https://arxiv.org/abs/1708.05678
function jumpdistance(zz)
    hh = zeros(Int64, length(zz[1]))
    avjd = 0
    for i in eachindex(zz)[2:end]
        j0 = findfirst(x -> x !=  0, zz[i]- zz[i-1])
        if j0 ≠ nothing
            hh[j0] += 1
        end
        j = abs(sum(zz[i] - zz[i-1]))
        #  j > 2 && error("")
         if j == 0.0
            continue
         else
            avjd += 1
         end
    end
    avjd/length(zz), hh
end

avjd, hh = jumpdistance(zz)
println("Average squared jump distance: $(avjd)")



colobs = hh/maximum(hh) 
fig1 = Figure()
ax = Axis(fig1[1,1])
sct = scatter!(ax, obs, markersize = colobs*25.0 .+ 5.0,
color = colobs,
colormap = :thermal)
Colorbar(fig1[1, 2], sct)
hlines!(ax, μtrue, color = :red, linestyle = :dash)
hlines!(ax, [(μtrue[1] + μtrue[2])/2, (μtrue[2] + μtrue[3])/2] , color = :blue, linestyle = :dash)
save("./version_$(version)_scatterplot.pdf", fig1)


fig2 = Figure()
ax2 = Axis(fig2[1,1])
for i in eachindex(mm[1])
    lines!(ax2, getindex.(mm, i))
end
hlines!(ax2, μtrue, color = (:red, 0.1), linestyle = :dash)
save("./version_$(version)_trace_means.pdf", fig2)
