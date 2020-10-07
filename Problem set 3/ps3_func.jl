# Emmanuel Murray Leclair
# Functions, problem set 3

# Function to find polynomial coefficients with monomial basis (global interpolation)
function mono_global(xfit::Array{Float64,1},yfit::Array{Float64,1})
    # xfit : interpolation points
    # yfit : value of the true function V(x) for each x=xfit
    # This function finds polynomial coefficients such that ̃V(xfit)=V(xfit) ∀ xfit
    m = length(xfit) # number of grid points to match and degree of polynomial
    A = zeros((m,m)) # initialize basis matrix
    a = zeros((m,1)) # initizlize polynomial coefficient array
    phi(m,x) = x^m # monomial basis
    for i in 1:m
        for j in 1:m
            A[j,i] = phi(i-1,xfit[j])
        end
    end
    a = inv(A)*yfit
    return a
end

# Function that find optimal curvature θ of grid to maximize interpolation accuracy using bisection
function optimal_gridcurv(a,b,N,F::Function,T::Function)
    # (a,b): range of interpolation
    # N: size of grid
    # F: function to interpolate
    # T: interpolation method
    grid(θ) = a.+ (b-a).*(range(0,1,length=N).^θ) # function that maps a (0,1) grid of length n to a (a,b) polynomial grid of length n using polynomial scaling
    xaxis = range(a,b;length=1000); # range to evaluate accuracy of inerpolation
    V = F(collect(xaxis)) # value of the function in all range
    θ_lb = 1 # lower bound on curvature (no curvature)
    θ_ub = 15 # upper bound on curvature
    w = 1.61803398875 # Golden ratio
    # two new points within (θ_lb,θ_ub) according to the golden ratio
    aa = θ_ub-(θ_ub-θ_lb)/w
    bb = θ_lb+(θ_ub-θ_lb)/w

    # Interpolation accuracy at lower bound
    grid_lb = grid(θ_lb)
    V_grid_lb = F(collect(grid_lb))
    Vtilde_lb = T(V_grid_lb,grid_lb,xaxis)
    V_diff_lb = findmax(abs.(((Vtilde_lb.-V)./V).*100))[1]
    # Interpolation accuracy at upper bound
    grid_ub = grid(θ_ub)
    V_grid_ub = F(collect(grid_ub))
    Vtilde_ub = T(V_grid_ub,grid_ub,xaxis)
    V_diff_ub = findmax(abs.(((Vtilde_ub.-V)./V).*100))[1]
    # Interpolation accuracy at both new points
    grid_aa = grid(aa)
    V_grid_aa = F(collect(grid_aa))
    Vtilde_aa = T(V_grid_aa,grid_aa,xaxis)
    V_diff_aa = findmax(abs.(((Vtilde_aa.-V)./V).*100))[1]
    grid_bb = grid(bb)
    V_grid_bb = F(collect(grid_bb))
    Vtilde_bb = T(V_grid_bb,grid_bb,xaxis)
    V_diff_bb = findmax(abs.(((Vtilde_bb.-V)./V).*100))[1]
    # Check condition for a local minimum: f(θ_lb),f(θ_ub) > f(aa),f(bb)
    println("V_diff_lb:$V_diff_lb, V_diff_ub:$V_diff_ub, V_diff_aa:$V_diff_aa, V_diff_bb:$V_diff_bb")
    if V_diff_lb < V_diff_aa || V_diff_lb < V_diff_bb || V_diff_ub < V_diff_aa || V_diff_ub < V_diff_bb
        println("No local minimum within boundary ($θ_lb,$θ_ub)")
        return
    end

    # If there is a local minimum, then proceed with bisection
    nmax = 1500 # maximum number of iterations
    tol = 10^-5 # tolerance level
    θ = Array{Float64}(undef, 1)  # intialize curvature
    V_diff = Array{Float64}(undef, 1) # initialize difference
    Vtilde = Array{Float64}(undef, length(xaxis))
    gridd = Array{Float64}(undef, N) # initialize grid points
    V_grid = Array{Float64}(undef,N) # Initialize function at grid points
    n_iter = 1 # initizlize iteration
    while n_iter < nmax
        # find new points following golden rule
        aa = θ_ub-(θ_ub-θ_lb)/w
        bb = θ_lb+(θ_ub-θ_lb)/w
        grid_aa = grid(aa) # Grid with curvature aa
        grid_bb = grid(bb) # Grid with curvature bb
        V_grid_aa = F(collect(grid_aa)) # Value of the function at the grid points with curvature θ=aa
        V_grid_bb = F(collect(grid_bb))
        Vtilde_aa = T(V_grid_aa,grid_aa,xaxis) # value of the interpolated function in all ranges with curvature θ=aa
        Vtilde_bb = T(V_grid_bb,grid_bb,xaxis)
        V_diff_aa = findmax(abs.(((Vtilde_aa.-V)./V).*100))[1] # Interpolation accuracy using sup norm with curvature θ=aa
        V_diff_bb = findmax(abs.(((Vtilde_bb.-V)./V).*100))[1] # Interpolation accuracy using sup norm with curvature θ=aa
        # Compare f(aa) and f(bb)
        if V_diff_aa > V_diff_bb
            θ_lb = aa
            θ = bb
            V_diff = V_diff_bb
            Vtilde = Vtilde_bb
            gridd = grid_bb
            V_grid = V_grid_bb
        elseif V_diff_aa < V_diff_bb
            θ_ub = bb
            θ = aa
            V_diff = V_diff_aa
            Vtilde = Vtilde_aa
            gridd = grid_aa
            V_grid = V_grid_aa
        end
        println("current iteration : (iter: $n_iter, theta: $θ, V_diff: $V_diff, |θ_ub-θ_lb|:$(θ_ub-θ_lb)")
        # Check if converged
        if(abs(θ_ub-θ_lb) <= tol)
            break # stop iteration if convergence is met
        end
        # If it didn't converge, go to the next iteration using bisection
        n_iter = n_iter + 1 # current iteration
    end
    return (θ, V_diff, abs(θ_ub-θ_lb), n_iter, Vtilde, gridd, V_grid)
end
