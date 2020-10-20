using Random, Distributions
using Parameters
using Statistics
using Plots
using LinearAlgebra
using Latexify
using PrettyTables

# Set seed
Random.seed!(420)

# Paramters
    # Generate structure for parameters using Parameters module
    # Set default values for parameters
    @with_kw struct Par
        # Model Parameters
        ρ::Float64 = 0.9    ; # AR(1) persistence
        σ::Float64 = 2.0    ; # Standard deviation of inovations η
        n_draw::Int64 = 10000 ; # Number of draws for simulated markov chain
    end
# Allocate parameters to object p for future calling
p = Par()

# Function to generate markov chain z using monte carlo simulation
function Simulate_MC(p::Par)
    @unpack ρ,σ,n_draw = p
    # Generate inovations η
    η = rand(Normal(0.0,σ),n_draw)
    # initialize process for z
    z = zeros(n_draw)
    # Compute z according to AR(1) process
    z[1] = η[1]
    for i in 2:n_draw
        z[i] = ρ*z[i-1] + η[i]
    end
    return z
end

# Function to distretize AR(1) markov process with Tauchen (1986)
function Tauchen86(N,p::Par,Ω::Any=3)
    @unpack ρ,σ = p
    # INPUTS:
        # ρ: persistence of unerlying AR(1) process where z' = ρz+η
        # σ: Std dev of inovation η in AR(1) process where η∼N(0,σ^2)
        # N: Size of grid for discrete process
        # Ω: Denotes boundaries of the grid as the number of standard deviations below and above the mean (lb=-Ωσ,ub=Ωσ)
    # OUTPUT:
        # z: All possible values of discretized AR(1) process, equally spaced grid of size N
        # Π: Matrix of transition probabilities
        # PDF_z: Stationary PDF of z
    #--------------------------------------------------------------------------
     z = collect(range(-Ω*σ,Ω*σ;length=N))
     Δz= 2*Ω*σ/(N-1)
     # initialize and compute probability transition matrix
     Π = zeros(N,N)
     F(x) = cdf.(Normal(),x)
     function prob_ij(zzp,zz,z::Vector)
         if zzp == z[1]
             return F((zzp+Δz/2-ρ*zz)/σ)
         elseif zzp == z[N]
             return 1-F((zzp-Δz/2-ρ*zz)/σ)
         else
             return F((zzp+Δz/2-ρ*zz)/σ)-F((zzp-Δz/2-ρ*zz)/σ)
         end
     end
     for i in 1:N
         for j in 1:N
            Π[i,j] = prob_ij(z[j],z[i],z)
        end
     end
     # Stationary distribution
     PDF_z = real(eigvecs(Π')[:,end]); PDF_z = PDF_z/sum(PDF_z)
     return (z,Π,PDF_z)
end

# Function to distretize AR(1) markov process with Rouwenhorst (1995)
function Rouwenhorst95(N,p::Par)
    @unpack ρ,σ=p
    # INPUTS:
        # ρ: persistence of unerlying AR(1) process where z' = ρz+η
        # σ: Std dev of inovation η in AR(1) process where η∼N(0,σ^2)
        # N: Size of grid for discrete process
    # OUTPUT:
        # z: All possible values of discretized AR(1) process, equally spaced grid of size N
        # Π: Matrix of transition probabilities
        # PDF_z: Stationary PDF of z
    #---------------------------------------------------------------------------
    Π = zeros(N,N)
    Π_Nm = zeros(N-1,N-1)
    P = (1+ρ)/2
    ϕ = σ*(sqrt((N-1)/(1-ρ^2)))
    z = range(-ϕ,ϕ;length=N)
    if N==2
        Π = [P 1-P;1-P P]
    else
        Π_Nm = Rouwenhorst95(N-1,p)[2]
        o = zeros(N-1)
        Π = P*[Π_Nm o; o' 0] + (1-P)*[o Π_Nm; 0 o'] + (1-P)*[o' 0; Π_Nm o] + P*[0 o';o Π_Nm]
        Π = Π./repeat(sum(Π,dims=2),1,N)
    end
    PDF_z = pdf.(Binomial(N-1,0.5),(0:N-1))
    return (z,Π,PDF_z)
end

# Function to simulate a discrete markov process
function Simulate_discrete(N,z,PDF_s,Π,p::Par)
    @unpack ρ,σ,n_draw = p
    # INPUT:
        # n_draw: Number of draws for the simulation
        # N     : Number of possible values for the discrete process
        # z     : Vector of possible values for the discrete process
        # PDF_s : Stationary PDF (used to draw initial point)
        # Π     : Transition probability matrix
    # OUTPUT:
        # z_MC  : Simulated draws
    # Initialization
    z_ind = zeros(Int64,n_draw) # index values of z
    z_MC = zeros(n_draw)
    # Initial value
    z_ind[1] = rand(Categorical(PDF_s))
    z_MC[1] = z[z_ind[1]]
    # Simulation
    for i in 2:n_draw
        z_ind[i] = rand(Categorical(Π[z_ind[i-1],:]))
        z_MC[i] = z[z_ind[i]]
    end
    return z_MC
end

# Function that calculate numerical moments out of a given array
function num_moments(z)
    # First four moments
    num_mean = mean(z)
    num_var = var(z)
    num_skew = skewness(z)
    num_kurt = kurtosis(z)
    # Autocorrelation going for t-1 to t-4
    autocorr = zeros(4)
    for i in 1:4
        autocorr[i] = cor(z[(i+1):end],z[1:(end-i)])
    end
    return (num_mean,num_var,num_skew,num_kurt,autocorr)
end


                    ### a) ###
# Simulate markov chain
z = Simulate_MC(p)
# Plot markov chain
zaxis = range(1,p.n_draw;length=p.n_draw)
gr()
plot(zaxis,z,label=nothing)

                    ### b) ###
N = [5,10]
# Tauchen (1986)
ztauc_MC = zeros(p.n_draw,2)
for i in 1:2
    (ztauc,Π_tauc,PDF_ztauc) = Tauchen86(N[i],p)
    ztauc_MC[:,i] = Simulate_discrete(N[i],ztauc,PDF_ztauc,Π_tauc,p)
end
# Rouwenhorst (1995)
zrouw_MC = zeros(p.n_draw,2)
for i in 1:2
    (zrouw,Π_rouw,PDF_zrouw) = Rouwenhorst95(N[i],p)
    zrouw_MC[:,i] = Simulate_discrete(N[i],zrouw,PDF_zrouw,Π_rouw,p)
end

                    ### c) ###
# Report moments for simulations, Ns=10000
tmean,rmean,tvar,rvar,tskew,rskew,tkur,rkur = [zeros(2) for i in 1:8]
mcmean,mcvar,mcskew,mckur = [zeros(1) for i in 1:4]
tautocorr,rautocorr = [zeros(4,2) for i in 1:2]
mcautocorr = zeros(4)
for i in 1:2
    (tmean[i],tvar[i],tskew[i],tkur[i],tautocorr[:,i]) = num_moments(ztauc_MC[:,i])
    (rmean[i],rvar[i],rskew[i],rkur[i],rautocorr[:,i]) = num_moments(zrouw_MC[:,i])
end
(mcmean,mcvar,mcskew,mckur,mcautocorr) = num_moments(z)
r(x) = round(x,digits=3)
Moments_Mat = [" " "Data" "Tauchen" "Tauchen" " " "Rouwenhorst" "Rouwenhorst";
               " " " " "N=5" "N=10" " " "N=5" "N=10" ;
               "mean" r(mcmean) r(tmean[1]) r(tmean[2]) " " r(rmean[1]) r(rmean[2]);
               "var"  r(mcvar)  r(tvar[1]) r(tvar[2]) " " r(rvar[1]) r(rvar[2]);
               "skew"  r(mcskew)  r(tskew[1]) r(tskew[2]) " " r(rskew[1]) r(rskew[2]);
               "kurt" r(mckur) r(tkur[1]) r(tkur[2]) " " r(rkur[1]) r(rkur[2]);
               "1st order acorr" r(mcautocorr[1]) r(tautocorr[1,1]) r(tautocorr[1,2]) " " r(rautocorr[1,1]) r(rautocorr[1,2]);
               "2nd order acorr" r(mcautocorr[2]) r(tautocorr[2,1]) r(tautocorr[2,2]) " " r(rautocorr[2,1]) r(rautocorr[2,2]);
               "3rd order acorr" r(mcautocorr[3]) r(tautocorr[3,1]) r(tautocorr[3,2]) " " r(rautocorr[3,1]) r(rautocorr[3,2]);
               "4th order acorr" r(mcautocorr[4]) r(tautocorr[4,1]) r(tautocorr[4,2]) " " r(rautocorr[4,1]) r(rautocorr[4,2])]

println("\n Moments from simulated paths: \n")
display(Moments_Mat)
pretty_table(Moments_Mat,noheader=true,backend = :latex)
