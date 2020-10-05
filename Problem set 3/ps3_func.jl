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

# Function that find optimal curvature θ of grid to maximize interpolation accuracy
#function optimal_gridcurv(a,b,N,F,T)
    # (a,b): range of interpolation
    # N: size of grid
    # F: function to interpolate
    # T: interpolation method
#end
