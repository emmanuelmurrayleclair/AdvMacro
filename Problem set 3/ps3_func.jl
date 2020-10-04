# Emmanuel Murray Leclair
# Functions, problem set 3

# Function for global interpolation with Monomial basis
function mono_global(xfit::Array{Float64,1},yfit::Array{Float64,1})
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
