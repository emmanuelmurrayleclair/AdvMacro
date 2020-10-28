using Random, Distributions
using Parameters
using Statistics
using Plots
using LinearAlgebra
using Latexify
using PrettyTables
using Interpolations
using Dierckx
using ForwardDiff
using Optim
     using Optim: converged, maximum, maximizer, minimizer, iterations
using Roots
include("Scaled_Interpolation_Functions.jl")

# Set seed
Random.seed!(420)

# Parameters
    # Generate structure for parameters using Parameters module
    # Set default values for parameters
    @with_kw struct Par
        # Model Parameters
        z_bar::Float64 = 1; # Reference level for productivity
        α::Float64 = 1/3  ; # Production function
        β::Float64 = 0.98 ; # Discount factor
        σ::Float64 = 2 ; # consumption elasticity of subsitution
        η::Float64 = 1 ; # labor/leisure elasticity of substitution
        δ::Float64 = 0.05 ; # Depreciation rate of capital
        ρ::Float64 = 0.9 ; # Persistance of AR(1) productivity process: log(z')=ρlog(z)+η
        σ_η::Float64 = 0.1 ; # Variance of innovations for productivity where η ∼ N(0,σ_η)
        # VFI Paramters
        max_iter::Int64   = 2000  ; # Maximum number of iterations
        dist_tol::Float64 = 1E-9  ; # Tolerance for distance between current and previous value functions
        # Policy functions
        H_tol::Float64    = 1E-9  ; # Tolerance for policy function iteration
        N_H::Int64        = 20    ; # Maximum number of policy iterations
        # Minimum consumption for numerical optimization
        c_min::Float64    = 1E-16
    end
# Allocate parameters to object p for future calling
p = Par()
global gam = ((p.α*p.z_bar*p.β)/(1-(1-p.δ)*p.β))^(1/(1-p.α)) # Some constant

# Steady state values
function SS_values(p::Par)
    # This function takes in parameters and provides steady state values
    # Parameters: productivity (z), returns to scale (α), discount factor (β), rate of capital depreciation (δ)
    #             consumption elasticity of substitution (σ), labor/leisure elasticity of subsitution (η)
    # Output: values for capital, labor, production, consumption, rental rate, wage
    @unpack z_bar,α,β,δ = p
    l_ss = 0.4 # Labor
    k_ss = gam*l_ss # Capital
    y_ss = z_bar*(k_ss^α)*(l_ss^(1-α)) # Output
    c_ss = y_ss-δ*k_ss # Consumption
    w_ss = (1-α)*z_bar*(k_ss^α)*(l_ss^(-α)) # Wage = marginal product of labor
    r_ss = α*z_bar*(k_ss^(α-1))*(l_ss^(1-α)) # Rental rate of capital = marginal product of capital
    return k_ss,y_ss,c_ss,r_ss,w_ss,l_ss
end
# Test steady state function
k_ss,y_ss,c_ss,r_ss,w_ss,l_ss = SS_values(p)
println(" ")
println("------------------------")
println(" Steady State variables")
println("   Quantities: k = $k_ss; y = $y_ss; c = $c_ss;")
println("   Prices:     r = $r_ss; w = $w_ss;")
println("------------------------")
println(" ")
# Get χ such that steady state labor = 0.4
function get_chi(p::Par,l_ss,c_ss,k_ss)
    @unpack z_bar, α, β, δ, σ, η = p
    chi = (c_ss^(-σ))*z_bar*(1-α)*(k_ss^α)*(l_ss^(-α-η))
    return chi
end
global χ = get_chi(p,l_ss,c_ss,k_ss)

# Function to make grid for capital (first state var)
function Make_K_Grid(n_k,θ_k,p::Par)
    # Get SS
    k_ss,y_ss,c_ss,r_ss,w_ss,l_ss = SS_values(p)
    # Lower and upper bounds
    lb = 1E-5
    ub = 2*k_ss
    # Get k_grid
    if θ_k≠1
        k_grid = PolyRange(lb,ub;θ=θ_k,N=n_k)
    else
    k_grid = range(lb,ub,length=n_k)
    end
    # Return
    return k_grid
end

# Function that returns the percentage error in the euler equation
function Euler_Error(k,z,kp,kpp,l,lp,p::Par)
    # Return percentage error in Euler equation
    @unpack α, β, σ, δ = p
    LHS = (z.*(k.^α).*(l.^(1-α)).+(1-δ).*k.-kp).^(-σ)
    RHS = β.*(α.*z.*((lp./kp).^(1-α)).+(1-δ)).*((z.*(kp.^α).*(lp.^(1-α)).+(1-δ).*kp.-kpp).^(-σ))
    return real((RHS./LHS.-1).*100)
end

# Period utility function (no penalty)
function utility(k,z,kp,l,p::Par)
    @unpack α,δ,σ,η,c_min = p
    u_c = 0; # intialize utility from consumption
    # Utility of consumption
    c = max(z.*(k.^α)*(l.^(1-α)).+(1-δ).*k.-kp,c_min) # Consumption from resource constraint
    u_c = (c.^(1-σ))./(1-σ)
    # Disutility of labor
    u_l = χ.*((l.^(1+η))./(1+η))
    return u_c.-u_l
end
# Derivative of utility function wrt labor l
function d_utility_l(k,z,kp,l,p::Par)
    @unpack α,δ,σ,η,c_min = p
    c = z*(k^α)*(l^(1-α))+(1-δ)*k-kp
    d_u = 0
    if c>c_min
        d_u = c^(-σ)
    else
        d_u = c_min^(-σ)
    end
    return d_u*z*(k^α)*(1-α)*(l^(-α))-χ*(l^η)
end
# Derivative of utility function wrt capital k'
function d_utility_kp(k,z,kp,l,p::Par)
    @unpack α,δ,σ,η,c_min = p
    c = z*(k^α)*(l^(1-α))+(1-δ)*k-kp
    d_u = 0
    if c>c_min
        d_u = c^(-σ)
    else
        d_u = c_min^(-σ)
    end
    return -d_u
end

# Function to distretize AR(1) markov process with Rouwenhorst (1995)
function Rouwenhorst95(N,p::Par)
    @unpack ρ,σ_η=p
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
    ϕ = σ_η*(sqrt((N-1)/(1-ρ^2)))
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

# Generate structure of model objects
    @with_kw struct Model
        # Parameters
        p::Par = Par() # Model paramters
        # Grids
        θ_k::Float64    = 1     # Default Curvature of k_grid
        n_k::Int64      = 20    # Default Size of k_grid
        n_k_fine::Int64 = 1000  # Default Size of fine grid for interpolation
        n_z::Int64      = 10    # Default size of discretized grid for productivity as a markov process
        scale_type::Int64 = 1   # Default grid type (polynomial)
        k_grid          = Make_K_Grid(n_k,θ_k,p)    # k_grid for model solution
        k_grid_fine     = Make_K_Grid(n_k_fine,1,p) # Fine grid for interpolation
        # Value and policy functions
        V         = Array{Float64}(undef,n_k,n_z)   # Value Function
        G_kp      = Array{Float64}(undef,n_k,n_z)       # Policy Function for capital k'
        G_c       = Array{Float64}(undef,n_k,n_z)       # Policy Function for consumption c
        G_l       = Array{Float64}(undef,n_k,n_z)       # Policy Function for labor l
        V_fine    = Array{Float64}(undef,n_k_fine,n_z)  # Interpolated Value Function
        G_kp_fine = Array{Float64}(undef,n_k_fine,n_z)  # Interpolated Policy Function for capial k'
        G_c_fine  = Array{Float64}(undef,n_k_fine,n_z)  # Interpolated Policy Function for consumption c
        G_l_fine  = Array{Float64}(undef,n_k_fine,n_z)  # Interpolated Policy Function for labor l
        Euler     = Array{Float64}(undef,n_k_fine,n_z)  # Errors in Euler equation
        z_grid    = Array{Float64}(undef,n_z)       # discretized grid for productivity
        Π         = Array{Float64}(undef,n_z,n_z)   # Probability transition matrix for productivity
    end
    M = Model()

# Function that finds the fixed point of the value function, then interpolates between grid points
function VFI_fixedpoint(T::Function,M::Model)
            ### T : Bellman operator (interior loop) ###
            ### M : Model structure                  ###
    # Unpack model structure
    @unpack p,n_k,n_k_fine,θ_k,k_grid,k_grid_fine,n_z,V_fine,G_kp_fine,G_c_fine,G_l_fine,Euler,z_grid,Π = M
    # VFI paramters
    @unpack max_iter, dist_tol = p
    # Initialize variables for iteration
    V_old = zeros(n_k,n_z) # Value function with no savings
    for i in 1:n_z
        V_old[:,i] = map(x->utility(x,z_grid[i],0.0,l_ss,p),collect(k_grid))
    end
    V_dist = 1 # Distance between old and new value function
    iter = 1
    println(" ")
    println("------------------------")
    println("VFI - n_k=$n_k - grid curvature θ_k=$θ_k - n_z=$n_z")
    # Start VFI
    while iter <= max_iter
        # Update value function and policy functions
        a=1.0
        V_new = a*T(Model(M,V=copy(V_old)))[1] + (1-a)*V_old # Call Bellman operator which returns a new value function at each capital grid point
        # Update value function and distance between previous and current iteration
        V_dist = maximum(abs.(V_new./V_old.-1))
        V_old = V_new
        # Report progress
        println("   VFI Loop: iter=$iter, dist=",100*V_dist," %")
        # Report progress every 100 iterations
        #if mod(iter,100)==0
        #    println("   VFI Loop: iter=$iter, dist=",100*V_dist,"%")
        #end
        # Check if converged
        if V_dist <= dist_tol
            V_new, G_kp, G_c, G_l = T(Model(M,V=copy(V_new)))
            println("VFI - n_k=$n_k - θ_k=$θ_k")
            println("Converged after $iter iterations with a distance of ",100*V_dist," %")
            println("------------------------")
            println(" ")
            # Interpolate to fine grid on capital using natural cubic spline if it converged
            for i in 1:n_z
                V_ip = ScaledInterpolations(M.k_grid,V_new[:,i], FritschButlandMonotonicInterpolation()) # Monotonic spline because I know that the value function is always increasing in capital
                    V_fine[:,i] = V_ip.(collect(M.k_grid_fine))
                G_kp_ip = ScaledInterpolations(M.k_grid,G_kp[:,i], BSpline(Cubic(Line(OnGrid()))))
                    G_kp_fine[:,i] = G_kp_ip.(collect(M.k_grid_fine))
                G_c_ip = ScaledInterpolations(M.k_grid,G_c[:,i], BSpline(Cubic(Line(OnGrid()))))
                    G_c_fine[:,i] = G_c_ip.(collect(M.k_grid_fine))
                G_l_ip = ScaledInterpolations(M.k_grid,G_l[:,i], BSpline(Cubic(Line(OnGrid()))))
                    G_l_fine[:,i] = G_l_ip.(collect(M.k_grid_fine))
                # Percent Euler Error on fine grid
                #Euler[:,i] = Euler_Error(M.k_grid_fine,G_kp_fine,G_kp_ip.(collect(G_kp_fine)),G_l_fine,G_l_ip.(collect(G_kp_fine)),p)
            end
            # Update model with solution
            M = Model(M; V=V_new,G_kp=G_kp,G_c=G_c,G_l=G_l,V_fine=V_fine,G_kp_fine=G_kp_fine,G_c_fine=G_c_fine,G_l_fine=G_l_fine,Euler=Euler)
            return M
        end
        # If it didn't converge, go to next iteration
        iter += 1
    end
    # If it didn't converge, return error
    error("Error in VFI - Solution not found")
end

# Bellman operator for the nested continuous choice of labor and capital tomorrow with capital and productivity as state variables
function T_EGM(M::Model)
    @unpack p,n_k,k_grid,n_z,V,G_kp,G_c,G_l,z_grid,Π,z_grid = M
    @unpack β,α,δ,η,σ,c_min = p
    # Emax
    Emax = (β*Π*V')'
    # Interpolating Emax to take derivative
    Emax_ip = [x->ScaledInterpolations(k_grid,x[:,i], FritschButlandMonotonicInterpolation()) for i in 1:n_z]
    # Derivative of Emax
    dEmax(x,ind) = ForwardDiff.derivative(Emax_ip[ind](Emax),x)
    # Function that returns consumption ̃c(k',z) that solves k' FOC
    function ctilde(kp,ind)
        if dEmax(kp,ind) <= 0.0
            return c_min
        else
            return dEmax(kp,ind)^(-1/σ)
        end
    end
    #ctilde(kp,ind) = (dEmax(kp,ind))^(-1/σ)
    # Function that returns capital today k that satisfy labor l FOC
    ktilde(c,l,z) = (χ*(l^(η+α))/(z*(c^(-σ))*(1-α)))^(1/α)
    # Function that returns endogenous resource constraint to solve for labor numerically
    res_cons(l,c,z,kp) = z*(ktilde(c,l,z)^α)*(l^(1-α)) + (1-δ)*ktilde(c,l,z)-kp-c
    # Boundaries for labor
    l_min = 1E-16
    l_max = 1.0
    G_k = zeros(n_k,n_z)
    # Outer loop for grid of productivity today z
    for i in 1:n_z
        # Inner loop for grid of capital tomorrow k'
        for j in 1:n_k
            # Get consumption needed to satisfy k' FOC
            G_c[j,i] = ctilde(k_grid[j],i)
            # Find labor that satisfies the resource constraint
            min_result = optimize(x->res_cons(x,G_c[j,i],z_grid[i],k_grid[j]).^2,l_min,l_max,Brent())
            G_l[j,i] = min_result.minimizer
            # use optimal labor to find optimal capital today
            G_k[j,i] = ktilde(G_c[j,i],G_l[j,i],z_grid[i])
            # Update value function
            V[j,i] = utility(G_k[j,i],z_grid[i],k_grid[j],G_l[j,i],p) + Emax[j,i]
        end
        # Sort endogenous grid
        sort_id = sortperm(G_k[:,i])
        G_k[:,i] = G_k[:,i][sort_id]
        G_l[:,i] = G_l[:,i][sort_id]
        G_c[:,i] = G_c[:,i][sort_id]
        V[:,i] = V[:,i][sort_id]
    end

            ### FUNCTIONS USED FOR MANUAL OPTIMIZATION IF CAPITAL IS OUT OF BOUNDS ###
    # Boundaries on capital
    get_kp_max(k,l,z) = z*(k^α)*(l^(1-α)) + (1-δ)*k - c_min # Function because the max k' depends on (k,l,z)
    kp_min = 1.001*k_grid[1]
    # Function that returns objective function for a given (z,k,k',l)
    function Obj_fn(k,z,kp,l,Π_z::Vector,p::Par)
        # Π_z: Vector of conditional probabilites for productivity next period given z today
        Emax = sum(Π_z[x]*Vp[x](V).(kp) for x in 1:n_z) # sum of z_j probability when starting at z_i : ∑_j π_ij V(kp,z_j)
        return -utility(k,z,kp,l,p) - β*Emax
    end
    # Function that returns derivative of objective function wrt k'
    d_Obj_fn_kp(k,z,kp,l,Π_z::Vector,p::Par) = d_utility_kp(k,z,kp,l,p) + β*dVp(kp,Π_z)
    # Derivative of objective function wrt labor l
    d_Obj_fn_l(k,z,kp,l,p::Par) = d_utility_l(k,z,kp,l,p)
    # Define function that finds optimal labor l given (k,z,k') and returns objective function conditional on optimal labor
    function Obj_fn_condl(k,z,kp,Π_z::Vector,p::Par)
        # Check for corner solutions on labor
        dobj_min = d_utility_l(k,z,kp,l_min,p)
        dobj_max = d_utility_l(k,z,kp,l_max,p)
        if dobj_min <= 0
            return Obj_fn(k,z,kp,l_min,Π_z,p),l_min
        elseif dobj_max >= 0
            return Obj_fn(k,z,kp,l_max,Π_z,p),l_max
        else
        # if no corner solutions, find interior solution
            min_result = optimize(x->d_utility_l(k,z,kp,x,p).^2,l_min,l_max,Brent())
            l = min_result.minimizer
            return Obj_fn(k,z,kp,l,Π_z,p),l
        end
    end
    # Interpolate new value function and policy functions to exogenous grid
    V_new = zeros(n_k,n_z)
    G_c_new = zeros(n_k,n_z)
    G_l_new = zeros(n_k,n_z)
    for i in 1:n_z
        V_new[:,i] = Spline1D(G_k[:,i],V[:,i]).(collect(k_grid))
        G_l_new[:,i] = Spline1D(G_k[:,i],G_l[:,i]).(collect(k_grid))
        for ind = findall(<(l_min),G_l_new[:,i])
            G_l_new[ind,i] = l_min
        end
        for ind = findall(>(l_max),G_l_new[:,i])
            G_l_new[ind,i] = l_max
        end
        G_c_new[:,i] = Spline1D(G_k[:,i],G_c[:,i]).(collect(k_grid))
        for ind = findall(<(c_min),G_c_new[:,i])
            G_c_new[ind,i] = c_min
        end
        #G_kp[:,i] = max(z_grid[i].*(collect(k_grid).^(α)).*(G_l_new[:,i].^(1-α)).+(1-δ).*(collect(k_grid)).-G_c_new[:,i],kp_min)
        for j in 1:n_k
            G_kp[j,i] = max(z_grid[i]*(k_grid[j]^(α)).*(G_l_new[j,i]^(1-α))+(1-δ)*(k_grid[j])-G_c_new[j,i],kp_min)
        end
        ## Manual optimization if capital is out of bounds
        #for ind = vcat(findall(<(G_k[1,i]),collect(k_grid)),findall(>(G_k[end,i]),collect(k_grid)))
        #    kp_max = min(get_kp_max(k_grid[ind],1.0,z_grid[i]),k_grid[end])
        #    # Check for corner solutions on capital
        #    l_kp_min = Obj_fn_condl(k_grid[ind],z_grid[i],kp_min,Π[i,:],p)[2]
        #    l_kp_max = Obj_fn_condl(k_grid[ind],z_grid[i],kp_max,Π[i,:],p)[2]
        #    dobj_min = d_Obj_fn_kp(k_grid[ind],z_grid[i],kp_min,l_kp_min,Π[i,:],p)
        #    dobj_max = d_Obj_fn_kp(k_grid[ind],z_grid[i],kp_max,l_kp_max,Π[i,:],p)
        #    if dobj_min <= 0.0
        #        G_kp[ind,i] = kp_min
        #        G_l_new[ind,i] = l_kp_min
        #        V_new[ind,i] = -Obj_fn(k_grid[ind],z_grid[i],kp_min,l_kp_min,Π[i,:],p)
        #    elseif dobj_max >= 0.0
        #        G_kp[ind,i] = kp_max
        #        G_l[ind,i] = l_kp_max
        #        V_new[ind,i] = -Obj_fn(k_grid[ind],z_grid[i],kp_max,l_kp_max,Π[i,:],p)
        #    else
        #        # If no corner solution, find interior solution
        #        min_result = optimize(x->Obj_fn_condl(k_grid[ind],z_grid[i],x,Π[i,:],p)[1],kp_min,kp_max,Brent())
        #        # Record results
        #        V_new[ind,i] = -min_result.minimum
        #        G_kp[ind,i] = min_result.minimizer
        #        G_l_new[ind,i] = Obj_fn_condl(k_grid[ind],z_grid[i],G_kp[ind,i],Π[i,:],p)[2]
        #    end
        #end
    end
    # Return results
    V = V_new
    G_l = G_l_new
    G_c = G_c_new
    return V, G_kp, G_c, G_l
end


                ### Solve the problem for θ_k=2,n_z=10,n_k=20 ###
# Get discrete grid for productivity and transition matrix
(log_z,Π) = Rouwenhorst95(10,p)[1:2]
z = exp.(log_z)
# Get solution
@time Mc  = VFI_fixedpoint(T_EGM,Model(n_k=50,θ_k=2,n_z=10,z_grid=z,Π=Π))
# Interpolate the value function along the z dimension for 3d plot
z_grid = range(z[1],z[end],length=size(z)[1])
z_grid_fine = range(z[1],z[end],length=Mc.n_k_fine)
V_fine_3d = zeros(Mc.n_k_fine,Mc.n_k_fine)
V_fine_33d = [ScaledInterpolations(z_grid,Mc.V_fine[i,:], BSpline(Cubic(Line(OnGrid())))).(collect(z_grid_fine)) for i in 1:Mc.n_k_fine]
for i in 1:Mc.n_k_fine
    V_fine_3d[i,:] = V_fine_33d[i]
end
# Surface and contour plot of the value function
gr()
x= Mc.k_grid_fine; y=z_grid_fine; f=V_fine_3d;
plot(x,y,f, st=:surface,xlabel="capital",ylabel="productivity") # Surface plot
savefig("./Figures/surface_vf.pdf")
plot(x,y,f, st=:contour,xlabel="capital",ylabel="productivity",nlevels=100, width=2, size=[800,480]) # Contour plot
savefig("./Figures/contour_vf.pdf")


            ### Some tests ###

(log_z,Π) = Rouwenhorst95(10,p)[1:2]
z = exp.(log_z)
M = Model(n_k=50,θ_k=2,n_z=10,z_grid=z,Π=Π)
@unpack p,n_k,k_grid,n_z,V,G_kp,G_c,G_l,z_grid,Π,z_grid = M
@unpack β,α,δ,η,σ,c_min = p

### Some tests ###
V_old = zeros(n_k,n_z) # Value function with no savings
for i in 1:n_z
V_old[:,i] = map(x->utility(x,z_grid[i],0.0,l_ss,p),collect(k_grid))
end

V = V_old

# Emax
Emax_vec = (β*Π*V')'
# Interpolating Emax to take derivative
Emax_ip = [x->ScaledInterpolations(k_grid,x[:,i], FritschButlandMonotonicInterpolation()) for i in 1:n_z]
# Derivative of Emax
dEmax(x,ind) = ForwardDiff.derivative(Emax_ip[ind](Emax_vec),x)
# Function that returns consumption ̃c(k',z) that solves k' FOC
function ctilde(kp,ind)
    if dEmax(kp,ind) <= 0.0
        return c_min
    else
        return dEmax(kp,ind)^(-1/σ)
    end
end
# Function that returns capital today k that satisfy labor l FOC
ktilde(c,l,z) = (χ*(l^(η+α))/(z*(c^(-σ))*(1-α)))^(1/α)
# Function that returns endogenous resource constraint to solve for labor numerically
res_cons(l,c,z,kp) = z*(ktilde(c,l,z)^α)*(l^(1-α)) + (1-δ)*ktilde(c,l,z)-kp-c

# Boundaries for labor
l_min = 1E-16
l_max = 1.0
G_k = zeros(n_k,n_z)
# Outer loop for grid of productivity today z
for i in 1:n_z
    # Inner loop for grid of capital tomorrow k'
    for j in 1:n_k
        # Get consumption needed to satisfy k' FOC
        G_c[j,i] = ctilde(k_grid[j],i)
        # Find labor that satisfies the resource constraint
        min_result = optimize(x->res_cons(x,G_c[j,i],z_grid[i],k_grid[j]).^2,l_min,l_max,Brent())
        G_l[j,i] = min_result.minimizer
        # use optimal labor to find optimal capital today
        G_k[j,i] = ktilde(G_c[j,i],G_l[j,i],z_grid[i])
        # Update value function
        V[j,i] = utility(G_k[j,i],z_grid[i],k_grid[j],G_l[j,i],p) + Emax_vec[j,i]
    end
    # Sort endogenous grid
    sort_id = sortperm(G_k[:,i])
    G_k[:,i] = G_k[:,i][sort_id]
    G_l[:,i] = G_l[:,i][sort_id]
    G_c[:,i] = G_c[:,i][sort_id]
    V[:,i] = V[:,i][sort_id]
end


        ### FUNCTIONS USED FOR MANUAL OPTIMIZATION IF CAPITAL IS OUT OF BOUNDS ###
# Boundaries on capital
get_kp_max(k,l,z) = z*(k^α)*(l^(1-α)) + (1-δ)*k - c_min # Function because the max k' depends on (k,l,z)
kp_min = 1.001*k_grid[1]
# Function that returns objective function for a given (z,k,k',l)
function Obj_fn(k,z,kp,l,Π_z::Vector,p::Par)
    # Π_z: Vector of conditional probabilites for productivity next period given z today
    Emax = sum(Π_z[x]*Vp[x](V).(kp) for x in 1:n_z) # sum of z_j probability when starting at z_i : ∑_j π_ij V(kp,z_j)
    return -utility(k,z,kp,l,p) - β*Emax
end
# Function that returns derivative of objective function wrt k'
d_Obj_fn_kp(k,z,kp,l,Π_z::Vector,p::Par) = d_utility_kp(k,z,kp,l,p) + β*dVp(kp,Π_z)
# Derivative of objective function wrt labor l
d_Obj_fn_l(k,z,kp,l,p::Par) = d_utility_l(k,z,kp,l,p)
# Define function that finds optimal labor l given (k,z,k') and returns objective function conditional on optimal labor
function Obj_fn_condl(k,z,kp,Π_z::Vector,p::Par)
    # Check for corner solutions on labor
    dobj_min = d_utility_l(k,z,kp,l_min,p)
    dobj_max = d_utility_l(k,z,kp,l_max,p)
    if dobj_min <= 0
        return Obj_fn(k,z,kp,l_min,Π_z,p),l_min
    elseif dobj_max >= 0
        return Obj_fn(k,z,kp,l_max,Π_z,p),l_max
    else
    # if no corner solutions, find interior solution
        min_result = optimize(x->d_utility_l(k,z,kp,x,p).^2,l_min,l_max,Brent())
        l = min_result.minimizer
        return Obj_fn(k,z,kp,l,Π_z,p),l
    end
end

# Interpolate new value function and policy functions to exogenous grid
V_new = zeros(n_k,n_z)
G_c_new = zeros(n_k,n_z)
G_l_new = zeros(n_k,n_z)
for i in 1:n_z
    V_new[:,i] = Spline1D(G_k[:,i],V[:,i]).(collect(k_grid))
    G_l_new[:,i] = Spline1D(G_k[:,i],G_l[:,i]).(collect(k_grid))
    for ind = findall(<(l_min),G_l_new[:,i])
        G_l_new[ind,i] = l_min
    end
    for ind = findall(>(l_max),G_l_new[:,i])
        G_l_new[ind,i] = l_max
    end
    G_c_new[:,i] = Spline1D(G_k[:,i],G_c[:,i]).(collect(k_grid))
    for ind = findall(<(c_min),G_c_new[:,i])
        G_c_new[ind,i] = c_min
    end
    G_kp[:,i] = z_grid[i].*(collect(k_grid).^(α)).*(G_l_new[:,i].^(1-α)).+(1-δ).*(collect(k_grid)).-G_c_new[:,i]
    # Manual optimization if capital is out of bounds
    for ind = vcat(findall(<(G_k[1,i]),collect(k_grid)),findall(>(G_k[end,i]),collect(k_grid)))
        kp_max = min(get_kp_max(k_grid[ind],1.0,z_grid[i]),k_grid[end])
        # Check for corner solutions on capital
        l_kp_min = Obj_fn_condl(k_grid[ind],z_grid[i],kp_min,Π[i,:],p)[2]
        l_kp_max = Obj_fn_condl(k_grid[ind],z_grid[i],kp_max,Π[i,:],p)[2]
        dobj_min = d_Obj_fn_kp(k_grid[ind],z_grid[i],kp_min,l_kp_min,Π[i,:],p)
        dobj_max = d_Obj_fn_kp(k_grid[ind],z_grid[i],kp_max,l_kp_max,Π[i,:],p)
        if dobj_min <= 0.0
            G_kp[ind,i] = kp_min
            G_l_new[ind,i] = l_kp_min
            V_new[ind,i] = -Obj_fn(k_grid[ind],z_grid[i],kp_min,l_kp_min,Π[i,:],p)
        elseif dobj_max >= 0.0
            G_kp[ind,i] = kp_max
            G_l[ind,i] = l_kp_max
            V_new[ind,i] = -Obj_fn(k_grid[ind],z_grid[i],kp_max,l_kp_max,Π[i,:],p)
        else
            # If no corner solution, find interior solution
            min_result = optimize(x->Obj_fn_condl(k_grid[ind],z_grid[i],x,Π[i,:],p)[1],kp_min,kp_max,Brent())
            # Record results
            V_new[ind,i] = -min_result.minimum
            G_kp[ind,i] = min_result.minimizer
            G_l_new[ind,i] = Obj_fn_condl(k_grid[ind],z_grid[i],G_kp[ind,i],Π[i,:],p)[2]
        end
    end
end
# Return results
V = V_new
G_l = G_l_new
G_c = G_c_new
return V, G_kp, G_c, G_l
