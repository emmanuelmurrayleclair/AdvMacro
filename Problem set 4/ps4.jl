using Plots
using Parameters
using Interpolations
using Dierckx
using ForwardDiff
using Optim
     using Optim: converged, maximum, maximizer, minimizer, iterations
using Roots
include("Scaled_Interpolation_Functions.jl")

# defining parameters

# Paramters
    # Generate structure for parameters using Parameters module
    # Set default values for parameters
    @with_kw struct Par
        # Model Parameters
        z::Float64 = 1    ; # Productivity
        α::Float64 = 1/3  ; # Production function
        β::Float64 = 0.98 ; # Discount factor
        σ::Float64 = 2 ; # consumption elasticity of subsitution
        η::Float64 = 1 ; # labor/leisure elasticity of substitution
        δ::Float64 = 0.05 ; # Depreciation rate of capital
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
global gam = ((p.α*p.z*p.β)/(1-(1-p.δ)*p.β))^(1/(1-p.α)) # Some constant

# Steady state values
function SS_values(p::Par)
    # This function takes in parameters and provides steady state values
    # Parameters: productivity (z), returns to scale (α), discount factor (β), rate of capital depreciation (δ)
    #             consumption elasticity of substitution (σ), labor/leisure elasticity of subsitution (η)
    # Output: values for capital, labor, production, consumption, rental rate, wage
    @unpack z,α,β,δ = p
    l_ss = 0.4 # Labor
    k_ss = gam*l_ss # Capital
    y_ss = z*(k_ss^α)*(l_ss^(1-α)) # Output
    c_ss = y_ss-δ*k_ss # Consumption
    w_ss = (1-α)*z*(k_ss^α)*(l_ss^(-α)) # Wage = marginal product of labor
    r_ss = α*z*(k_ss^(α-1))*(l_ss^(1-α)) # Rental rate of capital = marginal product of capital
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
    @unpack z, α, β, δ, σ, η = p
    chi = (c_ss^(-σ))*z*(1-α)*(k_ss^α)*(l_ss^(-α-η))
    return chi
end
global χ = get_chi(p,l_ss,c_ss,k_ss)

# Function to make grid for capital
function Make_K_Grid(n_k,θ_k,p::Par)
    # Get SS
    k_ss,y_ss,c_ss,r_ss,w_ss,l_ss = SS_values(p)
    # Lower and upper bounds
    lb = 1E-5
    ub = 2*k_ss
    # Get k_grid
    if θ_k≠1
        k_grid = PolyRange(1E-5,2*k_ss;θ=θ_k,N=n_k)
    else
    k_grid = range(1E-5,2*k_ss,length=n_k)
    end
    # Return
    return k_grid
end

# Function that returns the percentage error in the euler equation
function Euler_Error(k,kp,kpp,l,lp,p::Par)
    # Return percentage error in Euler equation
    @unpack z, α, β, σ, δ = p
    LHS = (z.*(k.^α).*(l.^(1-α)).+(1-δ).*k.-kp).^(-σ)
    RHS = β.*(α.*z.*((lp./kp).^(1-α)).+(1-δ)).*((z.*(kp.^α).*(lp.^(1-α)).+(1-δ).*kp.-kpp).^(-σ))
    return (RHS./LHS.-1).*100
end

# Period utility function with penalty
function utility(k,kp,l,a,p::Par) # a is penalty coefficent for quadratic loss
    @unpack α,δ,σ,z,η,c_min = p
    c = z*(k^α)*(l^(1-α))+(1-δ)*k-kp # Consumption from resource constraint
    c_max = z*(k^α)*(l^(1-α))+(1-δ)*k # maximum amount of consumption allowable
    u_c = 0; # intialize utility from consumption
    # Utility of consumption
    if c<=c_min
        u_c = (c_min^(1-σ))/(1-σ) - a*((c-c_min)^2)
    elseif c>=c_max
        u_c = (c_max^(1-σ))/(1-σ) - a*((c-c_max)^2)
    else
        u_c = (c^(1-σ))/(1-σ)
    end
    # Disutility of labor
    l_min = 1E-16
    l_max = 1.0
    if l >= l_max
        u_l = χ*((l_max^(1+η))/(1+η)) + a*((l-l_max)^2) # penalty for being above l_max
    elseif l <= l_min
        u_l = χ*((l_min^(1+η))/(1+η)) + a*((l-l_min)^2) # penalty for being below l_min
    else
        u_l = χ*((l^(1+η))/(1+η))
    end
    return u_c-u_l
end

# Unrestricted utility function (no penalty)
function utility_unr(k,kp,l,p::Par)
    @unpack α,δ,σ,z,η,c_min = p
    u_c = 0; # intialize utility from consumption
    # Utility of consumption
    c = z*(k^α)*(l^(1-α))+(1-δ)*k-kp # Consumption from resource constraint
    u_c = (c^(1-σ))/(1-σ)
    # Disutility of labor
    u_l = χ*((l^(1+η))/(1+η))
    return u_c-u_l
end

# Derivative of utility (unrestricted)
function d_utility_kp0(k,kp,l,p::Par)
    return ForwardDiff.derivative(x->utility_unr(k,x,l,p),kp)
end
function d_utility_l0(k,kp,l,p::Par)
    return ForwardDiff.derivative(x->utility_unr(k,kp,x,p),l)
end

# Derivative of utility function wrt labor l
function d_utility_l(k,kp,l,p::Par)
    @unpack α,δ,σ,z,η,c_min = p
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
function d_utility_kp(k,kp,l,p::Par)
    @unpack α,δ,σ,z,η,c_min = p
    c = z*(k^α)*(l^(1-α))+(1-δ)*k-kp
    d_u = 0
    if c>c_min
        d_u = c^(-σ)
    else
        d_u = c_min^(-σ)
    end
    return -d_u
end

# Generate structure of model objects
    @with_kw struct Model
        # Parameters
        p::Par = Par() # Model paramters
        # Grids
        θ_k::Float64    = 1     # Default Curvature of k_grid
        n_k::Int64      = 20    # Default Size of k_grid
        n_k_fine::Int64 = 1000  # Default Size of fine grid for interpolation
        scale_type::Int64 = 1   # Default grid type (polynomial)
        k_grid          = Make_K_Grid(n_k,θ_k,p)    # k_grid for model solution
        k_grid_fine     = Make_K_Grid(n_k_fine,1,p) # Fine grid for interpolation
        # Value and policy functions
        V         = Array{Float64}(undef,n_k)       # Value Function
        G_kp      = Array{Float64}(undef,n_k)       # Policy Function for capital k'
        G_c       = Array{Float64}(undef,n_k)       # Policy Function for consumption c
        G_l       = Array{Float64}(undef,n_k)       # Policy Function for labor l
        V_fine    = Array{Float64}(undef,n_k_fine)  # Interpolated Value Function
        G_kp_fine = Array{Float64}(undef,n_k_fine)  # Interpolated Policy Function for capial k'
        G_c_fine  = Array{Float64}(undef,n_k_fine)  # Interpolated Policy Function for consumption c
        G_l_fine  = Array{Float64}(undef,n_k_fine)  # Interpolated Policy Function for labor l
        Euler     = Array{Float64}(undef,n_k_fine)  # Errors in Euler equation
    end
    M = Model()

# Graphs
function VFI_Graphs(M::Model,VFI_Type)
    gr()
    # Value Function
        plot(M.k_grid,M.V,linetype=:scatter,marker=(:diamond,4),markercolor=RGB(0.5,0.1,0.1),label="VFI - n_k=$(M.n_k) - θ_k=$(M.θ_k)")
        plot!(M.k_grid_fine,M.V_fine,linewidth=2.5,linestyle=(:dash),linecolor=RGB(0.4,0.4,0.4),label=nothing)
        xlabel!("Capital")
        ylabel!("Value")
        savefig("./Figures/VFI_"*VFI_Type*"_V_$(M.n_k)_$(M.θ_k).pdf")
    # Capital Policy Function
        plot(M.k_grid_fine,M.k_grid_fine,lw=1,linecolor=RGB(0.6,0.6,0.6),label=nothing)
        plot!(M.k_grid,M.G_kp,linetype=:scatter,marker=(:diamond,4),markercolor=RGB(0.5,0.1,0.1),label="VFI - n_k=$(M.n_k) - θ_k=$(M.θ_k)")
        plot!(M.k_grid_fine,M.G_kp_fine,linewidth=2.5,linestyle=(:dash),linecolor=RGB(0.4,0.4,0.4),label=nothing)
        xlabel!("Capital")
        ylabel!("Capital")
        savefig("./Figures/VFI_"*VFI_Type*"_G_kp_$(M.n_k)_$(M.θ_k).pdf")
    # Labor Policy Function
        plot(M.k_grid_fine,M.k_grid_fine,lw=1,linecolor=RGB(0.6,0.6,0.6),label=nothing)
        plot!(M.k_grid,M.G_l,linetype=:scatter,marker=(:diamond,4),markercolor=RGB(0.5,0.1,0.1),label="VFI - n_k=$(M.n_k) - θ_k=$(M.θ_k)")
        plot!(M.k_grid_fine,M.G_l_fine,linewidth=2.5,linestyle=(:dash),linecolor=RGB(0.4,0.4,0.4),label=nothing)
        xlabel!("Capital")
        ylabel!("labor")
        savefig("./Figures/VFI_"*VFI_Type*"_G_l_$(M.n_k)_$(M.θ_k).pdf")
    # Euler Percentage Error
        plot(M.k_grid_fine,zeros(M.n_k_fine),lw=1,linecolor=RGB(0.6,0.6,0.6),label=nothing,title = "Euler Equation Error (%)",legend=(0.75,0.2),foreground_color_legend = nothing,background_color_legend = nothing)
        plot!(M.k_grid_fine,M.Euler,linetype=:scatter,marker=(:diamond,2),linecolor=RGB(0.5,0.1,0.1),label="VFI - n_k=$(M.n_k) - θ_k=$(M.θ_k)")
        xlabel!("Capital")
        ylabel!("Percentage Points")
        savefig("./Figures/VFI_"*VFI_Type*"_Euler_$(M.n_k)_$(M.θ_k).pdf")
        println("\n     Graphs Completed for VFI_$VFI_Type - n_k=$(M.n_k) - θ_k=$(M.θ_k)\n")
end

# Function that finds the fixed point of the value function, then interpolates between grid points
function VFI_fixedpoint(T::Function,M::Model)
            ### T : Bellman operator (interior loop) ###
            ### M : Model structure                  ###
    # Unpack model structure
    @unpack p, n_k, n_k_fine, θ_k, k_grid, k_grid_fine = M
    # VFI paramters
    @unpack max_iter, dist_tol = p
    # Initialize variables for iteration
    V_old = zeros(n_k) # Value function
    V_dist = 1 # Distance between old and new value function
    iter = 1
    println(" ")
    println("------------------------")
    println("VFI - n_k=$n_k - grid curvature θ_k=$θ_k")
    # Start VFI
    while iter <= max_iter
        # Update value function and policy functions
        V_new, G_kp, G_c, G_l = T(Model(M,V=copy(V_old))) # Call Bellman operator which returns a new value function at each capital grid point
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
            println("VFI - n_k=$n_k - θ_k=$θ_k")
            println("Converged after $iter iterations with a distance of ",100*V_dist," %")
            println("------------------------")
            println(" ")
            # Interpolate to fine grid using natural cubic spline if it converged
            V_ip = ScaledInterpolations(M.k_grid,V_new, FritschButlandMonotonicInterpolation()) # Monotonic spline because I know that the value function is always increasing in capital
                V_fine = V_ip.(collect(M.k_grid_fine))
            G_kp_ip = ScaledInterpolations(M.k_grid,G_kp, BSpline(Cubic(Line(OnGrid()))))
                G_kp_fine = G_kp_ip.(collect(M.k_grid_fine))
            G_c_ip = ScaledInterpolations(M.k_grid,G_c, BSpline(Cubic(Line(OnGrid()))))
                G_c_fine = G_c_ip.(collect(M.k_grid_fine))
            G_l_ip = ScaledInterpolations(M.k_grid,G_l, BSpline(Cubic(Line(OnGrid()))))
                G_l_fine = G_l_ip.(collect(M.k_grid_fine))
            # Percent Euler Error on fine grid
            Euler = Euler_Error(M.k_grid_fine,G_kp_fine,G_kp_ip.(collect(G_kp_fine)),G_l_fine,G_l_ip.(collect(G_kp_fine)),p)
            #Euler = map(x->Euler_Error(x[1],x[2],x[3],x[4],x[5],p),[M.k_grid_fine,G_kp_fine,G_kp_ip.(collect(G_kp_fine)),G_l_fine,G_l_ip.(collect(G_kp_fine))])
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

# Bellman operator for the continuous choice of both labor and capital simultaneously that maximizes the current iteration of the value function
function T_mvariate_max(M::Model)
    @unpack p, n_k, k_grid, V, G_kp, G_c, G_l = M
    @unpack β,α,z,δ,c_min = p
    # Interpolate current iteration of value function v(k') so I can evaluate it at any k'
    Vp = ScaledInterpolations(k_grid,V, BSpline(Cubic(Line(OnGrid())))) # Impose monotonicity because I know it is increasing in capital
    # Define derivative of value function wrt capital k'
    #dVp(x) = ForwardDiff.derivative(Vp,x)
    # Define function that returns objective function for a given triple (k,k',l)
    function Obj_fn(k,kp,l,a,p::Par)
        if kp < k_grid[1]
            return -utility(k,kp,l,a,p) - β*Vp.(k_grid[1])
        elseif kp > k_grid[end]
            return -utility(k,kp,l,a,p) - β*Vp.(k_grid[end])
        else
            return -utility(k,kp,l,a,p) - β*Vp.(kp)
        end
    end
    #Obj_fn(k,kp,l,a,p::Par) = -utility(k,kp,l,a,p) - β*Vp.(kp)
    # Derivative of objective function wrt capital k'
    #d_Obj_fn_kp(k,kp,l,p::Par) = d_utility_kp(k,kp,l,p) + β*dVp.(kp)
    # Derivative of objective function wrt labor l
    #d_Obj_fn_l(k,kp,l,p::Par) = d_utility_l(k,kp,l,p)
    # Define boundaries on capital tomorrow k' and on labor l
    l_min = 1E-16
    l_max = 1.0
    kp_min = 1.001*k_grid[1]
    #kp_min = 0.0001
    get_kp_max(k,l) = z*(k^α)*(l^(1-α)) + (1-δ)*k - c_min # Function because the max k' depends on the value of capital today k and labor l
    # Multivariate optimization of the objective function for each capital level in the grid
    # Steady state
    k_ss,l_ss = SS_values(p)
    # Initialization
    #V_new = zeros(n_k)
    kp_max = 0.0
    # Quadratic loss penalty
    a = 5.0
    # Maximize
    for i in 1:n_k
        #kp_max = min(get_kp_max(k_grid[i],1.0),0.9999*k_grid[end])
        kp_max = min(get_kp_max(k_grid[i],1.0),k_grid[end])
        # Initial values
        init_val = [(kp_min+kp_max)/2,l_ss]
        inner_optimizer = NelderMead()
        min_result = optimize(x->Obj_fn(k_grid[i],x[1],x[2],a,p),[kp_min,l_min],[kp_max,l_max],init_val,Fminbox(inner_optimizer))
        # Check resultHi
        #converged(min_result) || error("Failed to solve Bellman max for capital =" k_grid[i]" in $(iterations(min_result)) iterations")
        # Record results
        V[i] = -min_result.minimum
        (G_kp[i],G_l[i]) = min_result.minimizer
    end
    # Fill in policy for consumption
    G_c = z.*(collect(k_grid).^α).*(G_l.^(1-α)) .+(1-δ).*collect(k_grid) .-G_kp
    # Return results
    return V, G_kp, G_c, G_l
end

# Bellman operator for the continuous choice capital only where labor is solved for with the intratemporal FOC with bisection
function T_univariate_max(M::Model)
    @unpack p, n_k, k_grid, V, G_kp, G_c, G_l = M
    @unpack β,α,z,δ,η,σ,c_min = p
    # Interpolate current iteration of value function v(k') so I can evaluate it at any k'
    Vp = ScaledInterpolations(k_grid,V, BSpline(Cubic(Line(OnGrid())))) # Impose monotonicity because I know it is increasing in capital
    # Define derivative of value function wrt capital k'
    dVp(x) = ForwardDiff.derivative(Vp,x)
    # Define function that returns objective function for a given triple (k,k',l)
    Obj_fn(k,kp,l,p::Par) = -utility_unr(k,kp,l,p) - β*Vp.(kp)
    # Derivative of objective function wrt capital k'
    d_Obj_fn_kp(k,kp,l,p::Par) = d_utility_kp(k,kp,l,p) + β*dVp.(kp)
    # Derivative of objective function wrt labor l
    d_Obj_fn_l(k,kp,l,p::Par) = d_utility_l(k,kp,l,p)
    # Define boundaries on capital tomorrow k' and on labor l
    l_min = 1E-16
    l_max = 1.0
    get_kp_max(k,l) = z*(k^α)*(l^(1-α)) + (1-δ)*k - c_min # Function because the max k' depends on the value of capital today k and labor l
    # Define function that finds optimal labor l given (k,k') and returns objective function conditional on optimal labor
    function Obj_fn_condl(k,kp,p::Par)
        # Check for corner solutions
        dobj_min = d_utility_l(k,kp,l_min,p)
        dobj_max = d_utility_l(k,kp,l_max,p)
        if dobj_min <= 0
            return -utility_unr(k,kp,l_min,p) - β*Vp.(kp),l_min
        elseif dobj_max >= 0
            return -utility_unr(k,kp,l_max,p) - β*Vp.(kp),l_max
        else
        # if no corner solutions, find interior solution
            min_result = optimize(x->d_utility_l(k,kp,x,p).^2,l_min,l_max,GoldenSection())
            l = min_result.minimizer
            return -utility_unr(k,kp,l,p) - β*Vp.(kp),l
        end
    end
    # optimization of the objective function for each capital level in the grid
    # Initialization
    #min = 0.0001
    kp_min = 1.001*k_grid[1]
    # Maximize
    for i in 1:n_k
        #kp_max = min(get_kp_max(k_grid[i],1.0),0.9999*k_grid[end])
        kp_max = min(get_kp_max(k_grid[i],1.0),k_grid[end])
        # Check for corner solutions
        l_kp_min = Obj_fn_condl(k_grid[i],kp_min,p)[2]
        l_kp_max = Obj_fn_condl(k_grid[i],kp_max,p)[2]
        dobj_min = d_Obj_fn_kp(k_grid[i],kp_min,l_kp_min,p)
        dobj_max = d_Obj_fn_kp(k_grid[i],kp_max,l_kp_max,p)
        if dobj_min <= 0.0
            G_kp[i] = kp_min
            G_l[i] = l_kp_min
            V[i] = -Obj_fn(k_grid[i],kp_min,l_kp_min,p)
        elseif dobj_max >= 0.0
            G_kp[i] = kp_max
            G_l[i] = l_kp_max
            V[i] = -Obj_fn(k_grid[i],kp_max,l_kp_max,p)
        else
        # If no corner solution, find interior solution
            min_result = optimize(x->Obj_fn_condl(k_grid[i],x,p)[1],kp_min,kp_max,Brent())
            # Check result
            #converged(min_result) || error("Failed to solve Bellman max for capital =" k_grid[i]" in $(iterations(min_result)) iterations")
            # Record results
            V[i] = -min_result.minimum
            G_kp[i] = min_result.minimizer
            G_l[i] = Obj_fn_condl(k_grid[i],G_kp[i],p)[2]
        end
    end
    # Fill in policy for consumption
    G_c = z.*(collect(k_grid).^α).*(G_l.^(1-α)) .- G_kp
    # Return results
    return V, G_kp, G_c, G_l
end

                ### Execute VFI and plot graphs ###

# a) Curved spaced grid with multivariate optimization
@time M_20c  = VFI_fixedpoint(T_mvariate_max,Model(n_k=20,θ_k=3.6))
     VFI_Graphs(M_20c,"mvariate_max")
# b) Curved spaced grid with univariate optimization
@time M_20c  = VFI_fixedpoint(T_univariate_max,Model(n_k=20,θ_k=3.6))
     VFI_Graphs(M_20c,"univariate_max")


# Optimization test
f(x,y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2
x0 = [0.0,0.0]
optimize(x->f(x[1],x[2]),x0)
