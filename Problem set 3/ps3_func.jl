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
    grid(θ) = a.+ (b-a).*(range(0,1,length=N).^θ) # function that maps a (0,1) grid of length n to a (a,b) grid of length n using polynomial scaling
    xaxis = range(a,b;length=1000); # range to evaluate accuracy of inerpolation
    V = F(collect(xaxis)) # value of the function in all range
    θ_lb = 1 # lower bound on curvature (no curvature)
    θ_ub = 4.5 # upper bound on curvature
    w = 0.38197 # golden ratio
    θ_md = θ_lb*(1-w)+θ_ub*w # Curvature between lower and upper bound
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
    # Interpolation accuracy in between
    grid_md = grid(θ_md)
    V_grid_md = F(collect(grid_md))
    Vtilde_md = T(V_grid_md,grid_md,xaxis)
    V_diff_md = findmax(abs.(((Vtilde_md.-V)./V).*100))[1]
    # Check condition for a local minimum: f(θ_lb) > f(θ_md) < f(θ_ub)
    println("V_diff_lb:$V_diff_lb, V_diff_ub:$V_diff_ub, V_diff_mb:$V_diff_md")
    if V_diff_lb < V_diff_md || V_diff_ub < V_diff_md
        println("No local minimum within boundary ($θ_lb,$θ_ub)")
        return
    end

    # If there is a local minimum, then proceed with bisection
    nmax = 1500 # maximum number of iterations
    tol = 10^-5 # tolerance level
    θ = Array{Float64}(undef, 1)  # intialize curvature
    V_diff = Array{Float64}(undef, 1) # initialize difference
    θ = 1 # intialize curvature to get it outside of the loop
    n_iter = 1 # initizlize iteration
    z = 1-2*w
    while n_iter < nmax
        # find new point following golden rule
        c = (θ_ub-θ_lb)*z + θ_md
        grid_c = grid(c) # Grid with curvature c
        V_grid_c = F(collect(grid_c)) # Value of the function at the grid points with curvature θ=c
        Vtilde_c = T(V_grid_c,grid_c,xaxis) # value of the interpolated function in all ranges with curvature θ=c
        V_diff = findmax(abs.(((Vtilde_c.-V)./V).*100))[1] # Interpolation accuracy using sup norm with curvature θ=c
        println("current iteration : (iter: $n_iter, theta: $c, V_diff: $V_diff, |c-θ_md|:$(c-θ_md)")
        # Check if converged
        if (V_diff <= tol) || (abs(c-θ_md) <= tol)
            θ = c
            break # stop iteration if convergence is met
        end
        # If it didn't converge, go to the next iteration using bisection
        n_iter = n_iter + 1 # current iteration
        #grid_lb = grid(θ_lb)
        #grid_ub = grid(θ_ub)
        grid_md = grid(θ_md)
        #V_grid_lb = F(collect(grid_lb))
        #V_grid_ub = F(collect(grid_ub))
        V_grid_md = F(collect(grid_md))
        #Vtilde_lb = T(V_grid_lb,grid_lb,xaxis)
        #Vtilde_ub = T(V_grid_ub,grid_ub,xaxis)
        Vtilde_md = T(V_grid_md,grid_md,xaxis)
        #V_diff_lb = findmax(abs.(((Vtilde_lb.-V)./V).*100))[1]
        #V_diff_ub = findmax(abs.(((Vtilde_ub.-V)./V).*100))[1]
        V_diff_md = findmax(abs.(((Vtilde_md.-V)./V).*100))[1]
        if V_diff > V_diff_md
            θ_ub = c
        elseif V_diff < V_diff_md
            θ_lb = θ_md
            θ_md = c
        end
    end
    θ = c
    return θ
end
