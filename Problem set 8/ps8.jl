# Emmanuel Murray LEclair
# November 2020
# Hugget economy (Problem set 8 - Question 1)
mkpath("Figures")
using SparseArrays
using Random, Distributions
using Statistics
using LinearAlgebra
using Plots
using Interpolations
using Dierckx
using ForwardDiff
using Optim
    using Optim: converged, maximum, maximizer, minimizer, iterations
using Roots
using Parameters
include("Scaled_Interpolation_Functions.jl")
# using .VFI_Toolbox
println(" ")
println("------------------------")
println("Hugget (1993) in Julia")
println("PWD: ",pwd())
println("Solve Hugget's model with EGM and several implementations of histogram method")
println("------------------------")
println(" ")


#-----------------------------------------------------------
#-----------------------------------------------------------
# Paramters and Model Structure
    # Generate structure for parameters using Parameters module
    # We can set default values for our parameters
    @with_kw struct Par
        # Model Parameters
        α::Float64 = 1/3  ; # Production function
        β::Float64 = 0.96 ; # Discount factor
        γ::Float64 = 2.0  ; # Relative risk aversion parameter
        δ::Float64 = 0.05 ; # Depreciation rate
        ρ::Float64 = 0.90 ; # Persistence of labor efficiency process
        σ::Float64 = 0.08 ; # Standard devaiation of labor efficiency innovation
        z_bar::Float64 = 1; # Reference level for productivity
        # VFI Paramters
        max_iter::Int64   = 100000; # Maximum number of iterations
        dist_tol::Float64 = 1E-6  ; # Tolerance for distance
        # Histogram iteration parameters
        Hist_max_iter     = 10000 ;
        Hist_tol          = 1E-5  ;
        # Histogram iteration parameters
        N_eq              = 1000  ;
        tol_eq            = 1E-6  ;
        # Minimum consumption for numerical optimization
        c_min::Float64    = 1E-16
    end

    # Allocate paramters to object p for future calling
    p = Par()

# Function to distretize AR(1) markov process with Rouwenhorst (1995)
function Rouwenhorst95(N,p::Par)
    @unpack ρ,σ=p
    # INPUTS:
        # ρ: persistence of unerlying AR(1) process where log(z') = ρlog(z)+η
        # σ_z: Std dev of inovation η in AR(1) process where η∼N(0,σ^2)
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
    return (z,Π,P)
end

# Function to get grid on assets
function Make_A_Grid(n_a,θ_a,a_min,a_max,p::Par)
    # Get k_grid
    if θ_a≠1
        a_grid = PolyRange(a_min,a_max;θ=θ_a,N=n_a)
    else
    a_grid = range(a_min,a_max,length=n_a)
    end
    # Return
    return a_grid
end

    # Generate structure of model objects
    @with_kw struct Model
        # Parameters
        p::Par = Par() # Model paramters in their own structure
        # Parameters for Asset Grid
        a_max::Float64  = 50                         # Default max node of a_grid
        θ_a::Float64    = 2.5                        # Default Curvature of a_grid
        n_a::Int64      = 200                        # Default Size of a_grid
        n_a_fine::Int64 = 1000                       # Default Size of fine grid for interpolation and distribution
        # Productivity process
        n_ϵ::Int64     = 11     # Default size of discretized grid for productivity as a markov process
        log_ϵ          = Rouwenhorst95(n_ϵ,p)[1]
        Π              = Rouwenhorst95(n_ϵ,p)[2]
        ϵ_grid         = exp.(log_ϵ)
        # Prices and aggregates
        r::Float64 = 0.05 # Initial guess for rate of return on assets
        # Lower bound on Asset Grid (borrowing limit)
        a_min = -ϵ_grid[1]/r
        # Capital grid
        a_grid = Make_A_Grid(n_a,θ_a,a_min,a_max,p)          # Grid on assets for model solution
        a_grid_fine = Make_A_Grid(n_a_fine,1,a_min,a_max,p)  # Fine grid on assets for interpolation
        # State matrices
        a_mat     = repeat(a_grid',n_ϵ,1)
        a_mat_fine= repeat(a_grid_fine',n_ϵ,1)
        ϵ_mat     = repeat(ϵ_grid,1,n_a)
        # Value and policy functions
        V         = Array{Float64}(undef,n_ϵ,n_a)       # Value Function
        G_ap      = Array{Float64}(undef,n_ϵ,n_a)       # Policy Function
        G_c       = Array{Float64}(undef,n_ϵ,n_a)       # Policy Function
        V_fine    = Array{Float64}(undef,n_ϵ,n_a_fine)  # Value Function on fine grid
        G_ap_fine = Array{Float64}(undef,n_ϵ,n_a_fine)  # Policy Function on fine grid
        G_c_fine  = Array{Float64}(undef,n_ϵ,n_a_fine)  # Policy Function on fine grid
        # Distribution
        Γ         = 1/(n_ϵ*n_a_fine)*ones(n_ϵ,n_a_fine) # Distribution (initiliazed to uniform)
        # Error in Euler equation
        Euler     = Array{Float64}(undef,n_ϵ,n_a_fine)  # Errors in Euler equation
    end

    # Allocate model to object M for future calling
    M = Model()

# Utility function
function utility(c,p::Par)
    if p.γ>1
    return (c).^(1-p.γ)/(1-p.γ)
    else
    return log.(c)
    end
end
function d_utility(c,p::Par)
    return (c).^(-p.γ)
end
function d_utility_inv(x,p::Par)
    return x.^(-1/p.γ)
end

# Get index for histogram method
function get_ind(a_grid,g_a)
    n = length(a_grid)
    if g_a > maximum(a_grid)
        return n
    elseif g_a < minimum(a_grid)
        return 1
    else
        return floor(((g_a-minimum(a_grid))/(maximum(a_grid)-minimum(a_grid)))*(n-1)+1)
    end
end


# PFI Fixed Point - Iterate over policy functions for a given guess of the interest rate r
function PFI_Fixed_Point(T::Function,M::Model,G_ap_old=nothing)
    # Unpack model structure
    @unpack p, n_ϵ, n_a, n_a_fine, θ_a, a_grid, a_grid_fine, r = M
    # PFI paramters
    @unpack max_iter, dist_tol = p
    # Initialize variables for loop
    if G_ap_old==nothing
        G_ap_old  = (1+r)*M.a_mat
    end
    G_dist = 1              ; # Initialize distance
    println(" ")
    println("------------------------")
    println("PFI - n_ϵ=$n_ϵ, n_a=$n_a - θ_a=$θ_a - r=$r")
    for iter=1:max_iter
        # Update value function
        G_ap_new, G_c = T(Model(M,G_ap=copy(G_ap_old)))
        # Update distance and iterations
        G_dist = sqrt(norm(G_ap_new-G_ap_old,2))
        # Update old function
        G_ap_old  = G_ap_new
        # Report progress every 250 iterations
        if mod(iter,250)==0
            println("   PFI Loop: iter=$iter, dist=",G_dist)
        end
        # Check convergence and return results
        if G_dist<=dist_tol
            println("PFI - n_ϵ=$n_ϵ, n_a=$n_a - θ_a=$θ_a - r=$r")
            println("Iterations = $iter and Distance = ",G_dist)
            println("------------------------")
            println(" ")
            # Interpolate to fine grid
            G_ap_fine = zeros(n_ϵ,n_a_fine)
            G_c_fine  = zeros(n_ϵ,n_a_fine)
            for i_ϵ=1:n_ϵ
            G_ap_ip = ScaledInterpolations(a_grid,G_ap_new[i_ϵ,:] , BSpline(Cubic(Line(OnGrid()))))
                G_ap_fine[i_ϵ,:].= G_ap_ip.(collect(a_grid_fine))
            G_c_ip  = ScaledInterpolations(a_grid,G_c[i_ϵ,:]  , BSpline(Cubic(Line(OnGrid()))))
                G_c_fine[i_ϵ,:] .= G_c_ip.(collect(a_grid_fine))
            end
            # Update model
            M = Model(M; G_ap=G_ap_new,G_c=G_c,G_ap_fine=G_ap_fine,G_c_fine=G_c_fine)
            # Return results
            return M
        end
    end
    # If loop ends there was no convergence -> Error!
    error("Error in PFI - Solution not found")
end

# Bellman operator - EGM - Iterate on Policy Functions
function T_EGM_G(M::Model)
    @unpack p, n_ϵ, n_a, G_ap, r, a_min, Π = M
    @unpack β = p
    # Define RHS of Euler equation for each (ϵ,a')
    # Rows are present ϵ and columns are tomorrow's a in fixed grid
    Euler_RHS = β*(1+r)*Π*d_utility( (1+r)*M.a_mat + M.ϵ_mat - G_ap , p )
    # Check Monotonicity
    if any( Euler_RHS.<0 )
        error("RHS must be monotone for EGM to work")
    end
    # Define consumption from Euler equation
    C_endo = max.(d_utility_inv(Euler_RHS,p),p.c_min)
    # Define endogenous grid on assets
    A_endo = (C_endo .+ M.a_mat - M.ϵ_mat)/(1+r)
    # Interpolate functions on exogenous grid
    G_c = Array{Float64}(undef,n_ϵ,n_a)
    for i_ϵ=1:n_ϵ
        # Sort A_endo for interpolation
        sort_ind = sortperm(A_endo[i_ϵ,:])
        A_aux    = A_endo[i_ϵ,:][sort_ind]
        C_aux    = C_endo[i_ϵ,:][sort_ind]
        # Check boundary condition
            # Ap(ϵ,a)=a_min for all a<min(A_aux)
            # Note that in that case C is linear between a_min and min(A_aux)
        if minimum(A_aux)>a_min
            a_vec = M.a_grid[M.a_grid.<minimum(A_aux)] # All assets today between a_min and min(A_aux)
            A_aux = [a_vec ; A_aux] # Append to existing grid of today's assets
            C_aux = [((1+r)*a_vec.+M.ϵ_grid[i_ϵ].-a_min) ; C_aux] # I can know consumption directly because all those assets imply a'=a_min
        end
        C_ip        = Spline1D(A_aux,C_aux;k=1)
        G_c[i_ϵ,:] .= C_ip.(M.a_grid) # interpolate consumption to exogenous grid of assets today
        #Ap_aux      = (1+r)*collect(M.a_grid) .+ M.ϵ_grid[i_ϵ] .- G_c[i_ϵ,:] # Find corresponding assets tomorrow
    end
    # Update policy function
    G_ap .= (1+r)*M.a_mat .+ M.ϵ_mat .- G_c
        # Adjust for numerical error
        for ind = findall(<=(1e-10),abs.(G_ap.-a_min))
            G_ap[ind] = a_min
            G_c[ind]  = (1+r)*M.a_mat[ind] + M.ϵ_mat[ind] - a_min
        end
        # Check for borrowing constraint
        if any( G_ap.<a_min )
            error("Borrowing Constraint Violated")
        end
    # Return Results
    return G_ap, G_c
end

# Value function (given policy function)
function Value_Function(M::Model)
    # Unpack model structure
    @unpack p, n_ϵ, n_a, n_a_fine, a_grid, a_grid_fine, r, G_ap, G_c, Π = M
    @unpack β, dist_tol = p

    # Compute value function with policy function iteration
    V      = zeros(n_ϵ,n_a)
    V_new  = zeros(n_ϵ,n_a)
    V_dist = 1
    U_mat  = utility(G_c,p)
    while V_dist>dist_tol
        for i_ϵ=1:n_ϵ
            Pr = Π[i_ϵ,:]'
            for i_a=1:n_a
                ap = G_ap[i_ϵ,i_a]
                Vp = zeros(n_ϵ)
                    for i_ϵp=1:n_ϵ
                        Vp_ip    = ScaledInterpolations(a_grid,V[i_ϵp,:], BSpline(Cubic(Line(OnGrid()))))
                        Vp[i_ϵp] = Vp_ip(ap)
                    end
                V_new[i_ϵ,i_a] = U_mat[i_ϵ,i_a] + β*(Pr*Vp)
            end
        end
        V_dist = maximum(abs.(V_new./V.-1))
        V      = V_new
    end
    # Interpolate to fine grid
    V_fine    = zeros(n_ϵ,n_a_fine)
    for i_ϵ=1:n_ϵ
        V_ip    = ScaledInterpolations(a_grid,V[i_ϵ,:], BSpline(Cubic(Line(OnGrid()))))
        V_fine[i_ϵ,:] .= V_ip.(collect(a_grid_fine))
    end
    # Update model
    M = Model(M; V=V,V_fine=V_fine)
    return M
end

# Compute stationary distribution with Histogram method
# Histogram method
function Histogram_Method_Loop(M::Model,N_H=nothing,Γ_0=nothing)
    @unpack a_min, p, n_ϵ, Π, n_a_fine, a_grid_fine, G_ap_fine = M
    @unpack Hist_max_iter, Hist_tol = p

    println("\n--------------------------------\nBegining Histogram Method with Loops")

    # Change max iter
    if N_H==nothing
        N_H = Hist_max_iter
    end

    # Initial distribution (uniorm in income shock and assets)
    if Γ_0==nothing
        Γ_0 = M.Γ
    end

    # Discretize distribution
    H_ind    = Array{Int64}(undef,n_ϵ,n_a_fine)
    H_weight = Array{Float64}(undef,n_ϵ,n_a_fine)
    a_max    = maximum(a_grid_fine)
    for i_ϵ=1:n_ϵ
    for i_a=1:n_a_fine
        H_ind[i_ϵ,i_a]    = get_ind(a_grid_fine,G_ap_fine[i_ϵ,i_a])
        # weight on node corresponding to H_ind
        if H_ind[i_ϵ,i_a] == n_a_fine || H_ind[i_ϵ,i_a] == 1 # Mass points above max or below min in the fine grid
            H_weight[i_ϵ,i_a] = 1
        else
            H_weight[i_ϵ,i_a] = 1-(G_ap_fine[i_ϵ,i_a]-a_grid_fine[H_ind[i_ϵ,i_a]])/(a_grid_fine[H_ind[i_ϵ,i_a]+1]-a_grid_fine[H_ind[i_ϵ,i_a]])
        end
    end
    end

    #Γ = zeros(n_ϵ,n_a_fine)
    ## Loop for updating histogram
    #H_dist = 1
    #for l = 1:N_H
    #    for i = 1:n_ϵ
    #        for j = 1:n_a_fine
    #            # all combinations of (ϵ,a) today that may lead to a_j tomorrow
    #            index = findall(x->x==j,H_ind)
    #            index_ =  findall(x->x==j,H_ind.+1)
    #            # from the above, take only ϵ for the income probability transition
    #            transindex = map(i->i[1], index)
    #            transindex_ = map(i->i[1], index_)
    #            # Update the distribution
    #            Γ[i,j] = Π[i,transindex]'*(H_weight[index].*Γ_0[index])+Π[i,transindex_]'*((-H_weight[index_].+1).*Γ_0[index_])
    #        end
    #    end
    #    # Update distance
    #    H_dist = maximum(abs.(Γ-Γ_0))
    #    # Update initial distribution
    #    Γ_0 .= Γ
    #    # Report progress
    #    println("   Histogram Loop: iter=$l, dist=$H_dist")
    #    #if mod(l,250)==0
    #    #    println("   Histogram Loop: iter=$l, dist=$H_dist")
    #    #end
    #    # Check convergence
    #    if H_dist<Hist_tol
    #        println("Histogram iteartion converged in iteration $i_H. H_dist=$H_dist\n--------------------------------\n")
    #        M = Model(M; Γ=Γ)
    #        return M
    #    end
    #end

    Γ = zeros(n_ϵ,n_a_fine)
    # Loop for updating histogram
    H_dist = 1
    for i_H=1:N_H
        # Update histogram
        Γ = zeros(n_ϵ,n_a_fine)
        for i_ϵ=1:n_ϵ # Current ϵ
        for i_a=1:n_a_fine # Current a
            i_ap = H_ind[i_ϵ,i_a]
            ω_ap = H_weight[i_ϵ,i_a]
            for i_ϵp=1:n_ϵ # Future ϵ
                Γ[i_ϵp,i_ap]   = Γ[i_ϵp,i_ap]   +    ω_ap *Π[i_ϵ,i_ϵp]*Γ_0[i_ϵ,i_a]
                if i_ap < n_a_fine
                    Γ[i_ϵp,i_ap+1] = Γ[i_ϵp,i_ap+1] + (1-ω_ap)*Π[i_ϵ,i_ϵp]*Γ_0[i_ϵ,i_a]
                end
            end
        end
        end
        # Update distance
        H_dist = maximum(abs.(Γ-Γ_0))
        # Update initial distribution
        Γ_0 .= Γ
        # Report progress
        if mod(i_H,50)==0
            println("   Histogram Loop: iter=$i_H, dist=$H_dist")
        end
        # Check convergence
        if H_dist<Hist_tol
            println("Histogram iteartion converged in iteration $i_H. H_dist=$H_dist\n--------------------------------\n")
            M = Model(M; Γ=Γ)
            return M
        end
    end

    # Return Results
    println("Histogram updated for $N_H iteartions. Current H_dist=$H_dist \n--------------------------------\n")
    M = Model(M; Γ=Γ_0)
    return M
end

# Do optimization then check market clearing
function market_clearing(r,M::Model)
    # Iterate over policy function given r
    M = PFI_Fixed_Point(T_EGM_G,Model(r=r))
    # Get value function
    M = Value_Function(M)
    # Find stationary distribution
    M = Histogram_Method_Loop(M)
    # Check for market clearing
    distance = sum(sum(M.G_ap_fine.*M.Γ,dims=2)).^2
    println("Squared distance from market clearing=$distance at intertest rate = $r")
    return distance
end

# Outer loop to find interest rate that clears market
function S_RCE(M::Model)
    @unpack p = M
    @unpack β = p
    # Maximum interest rate
    r_max = 1/β-1
    # Minimum interest rate
    r_min = 0
    # Find equilibrium interest rate
    min_result = optimize(x->market_clearing(x,M).^2,r_min,r_max)
    # Check result
    converged(min_result) || error("Failed to clear asset market in $(iterations(min_result)) iterations")
    # Upddate policy function
    r  = min_result.minimizer
    println("Equilibrium found in $(iterations(min_result)) iterations: r=$r")
    # Load equilibrium interest rate into model
    M = Model(M; r=r)
    # Compute policy function, value function and stationary distribution
    M = PFI_Fixed_Point(T_EGM_G,M)
    M = Value_Function(M)
    M = Histogram_Method_Loop(M)
    return M
end

# Function that makes 3-d plots of the value function, policy function and asset distributon
function plot_hugget(M::Model)
    # Interpolate the value function, policy function and asset distribution along the income shock dimension
    ϵ_grid_fine = range(M.ϵ_grid[1],M.ϵ_grid[end],length=M.n_a_fine)
    V_fine3d = zeros(M.n_a_fine,M.n_a_fine)
    G_ap_fine3d = zeros(M.n_a_fine,M.n_a_fine)
    Γ_fine3d = zeros(M.n_a_fine,M.n_a_fine)
    for i in 1:M.n_a_fine
        Γ_fine3d[:,i] = Spline1D(M.ϵ_grid,M.Γ[:,i];k=1).(collect(ϵ_grid_fine))
        V_fine3d[:,i] = Spline1D(M.ϵ_grid,M.V_fine[:,i];k=1).(collect(ϵ_grid_fine))
        G_ap_fine3d[:,i] = Spline1D(M.ϵ_grid,M.G_ap_fine[:,i];k=1).(collect(ϵ_grid_fine))
    end
    # Normalize distribution such that it sums to one (need because I interpolated)
    Γ_fine3d = Γ_fine3d./(sum(sum(Γ_fine3d)))
    # Plots of the value function, policy function and asset distributon
    gr()
    plot(ϵ_grid_fine,M.a_grid_fine,V_fine3d, st=:contour,xlabel="income",ylabel="assets",title="Value Function -hugget: n_ϵ=$(M.n_ϵ) - n_a=$(M.n_a) - θ_a=$(M.θ_a), eq r = $(M.r)") # Surface plot
    savefig("./Figures/surface_vf_hugget.pdf")
    plot(ϵ_grid_fine,M.a_grid_fine,G_ap_fine3d, st=:contour,xlabel="income",ylabel="assets",title="Asset policy -hugget: n_ϵ=$(M.n_ϵ) - n_a=$(M.n_a) - θ_a=$(M.θ_a), eq r = $(M.r)") # Surface plot
    savefig("./Figures/surface_policy_hugget.pdf")
    plot(ϵ_grid_fine,M.a_grid_fine,Γ_fine3d, st=:surface,xlabel="income",ylabel="assets",title="Distribution -hugget: n_ϵ=$(M.n_ϵ) - n_a=$(M.n_a) - θ_a=$(M.θ_a), eq r = $(M.r)") # Surface plot
    savefig("./Figures/surface_dist_hugget.pdf")
end

# Call PFI
M_hugget = S_RCE(Model(n_a=100,n_ϵ=11))
# Get plots
    plot_hugget(M_hugget)
