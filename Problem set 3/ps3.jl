# Emmanuel Murray Leclair
# Main script, problem set 3
mkpath("Figures")
using Plots
using LinearAlgebra
using PrettyTables
include("ps3_func.jl")

                            ### QUESTION_1 AND 3 ###

# Utility functions
function U_fn(x::Array{Float64,1},index::Int64,σ::Array)
    u_fn = Float64[]
    if index == 1
        u_fn = log.(x)
        return u_fn
    elseif index == 2
        u_fn = sqrt.(x)
        return u_fn
    elseif index > 2
        u_fn = (x.^(1-σ[index-2]))./(1-σ[index-2])
        return u_fn
    end
end
σ = [2,5,10]

# Range of interpolation
xaxis = range(0.05,2;length=1000);
xaxis_extrapol = range(2,2.5,length=250)
# Interpolation points to match
n = [5,10,20] # Number of points
n_grid = 3
inter_p(p) = collect(range(0.05,2;length=p)) # function that divides the range (xaxis) into equally spaced interpolation points
xi = Vector{Vector{Float64}}([inter_p(n[1]),inter_p(n[2]),inter_p(n[3])])

n_f = 5 # Number of functions to interpolate
for i in 1:n_f
    # y-grid is such that V(x) = ̃V(x) ∀ x=xi
    yi = Vector{Vector{Float64}}([U_fn(xi[1],i,σ),U_fn(xi[2],i,σ),U_fn(xi[3],i,σ)])
    # True utility function V(x) - will be used as benchmark for all interpolation methods
    V = U_fn(collect(xaxis),i,σ)

                ### GLOBAL POLYNOMIAL INTERPOL WITH MONOMIAL BASIS (M) ###
    # Get polynomial coefficients
    a = Vector{Vector{Float64}}([mono_global(xi[1],yi[1])])
    push!(a,mono_global(xi[2],yi[2]))
    push!(a,mono_global(xi[3],yi[3]))
    # ̃V(x) with monomial polynomial coefficients
    Vtilde(x,a) = sum(a[m]*(x.^(m-1)) for m=1:length(a))
    Vtilde_m = zeros(length(xaxis),n_grid)
    for j in 1:n_grid
        Vtilde_m[:,j] = Vtilde(xaxis,a[j])
    end
    # Plot the results
    gr()
    plt = plot(title="Interpolation n= (5,10,20) - Monomial Polynomial")
    plot!(xaxis,V,linewidth=3,color=:black,label = "True uitility function",legend=(0.75,0.75),foreground_color_legend = nothing,background_color_legend = nothing)
    plot!(xaxis,Vtilde_m[:,1],linestyle=:dash,linewidth=2,label="Interpolation - 5 points")
    plot!(xaxis,Vtilde_m[:,2],linestyle=:dot,linewidth=2,label="Interpolation - 10 points")
    plot!(xaxis,Vtilde_m[:,3],linestyle=:dashdot,linewidth=2,label="Interpolation - 20 points")
    savefig("./Figures/global_mononial/U_fn$i.pdf")

            ### LOCAL INTERPOL WITH LINEAR SPLINE (LS) ###
    ind_fn(xi,x) = findmax(sign.(xi.-x))[2]-1
    ind = Array{Int64}(undef,length(xaxis),n_grid)
    A_ls = zeros(length(xaxis),n_grid)
    Vtilde_ls = zeros(length(xaxis),n_grid)
    for j in 1:n_grid
        # Find indices for local interpolation
        ind[:,j] = map(x->ind_fn(xi[j],x),xaxis)
        # Find local weights (A(x) and B(x)=1-A(x))
        A_ls[:,j] = (xi[j][ind[:,j].+1].-xaxis)./(xi[j][ind[:,j].+1].-xi[j][ind[:,j]])
        # Find ̃V(x)
        Vtilde_ls[:,j] = A_ls[:,j].*yi[j][ind[:,j]].+ (-A_ls[:,j].+1).*yi[j][ind[:,j].+1]
    end
    # Plot the results
    gr()
    plt = plot(title="Interpolation n= (5,10,20) - Linear spline")
    plot!(xaxis,V,linewidth=3,color=:black,label = "True uitility function",legend=(0.75,0.75),foreground_color_legend = nothing,background_color_legend = nothing)
    plot!(xaxis,Vtilde_ls[:,1],linestyle=:dash,linewidth=2,label="Interpolation - 5 points")
    plot!(xaxis,Vtilde_ls[:,2],linestyle=:dot,linewidth=2,label="Interpolation - 10 points")
    plot!(xaxis,Vtilde_ls[:,3],linestyle=:dashdot,linewidth=2,label="Interpolation - 20 points")
    savefig("./Figures/local_linspline/U_fn$i.pdf")

            ### LOCAL INTERPOL CUBIC SPLINE (CS) ###
    ## Solve tridiagonal system (find y'' in Cy''=S)
    # Define functions for tridiagonal system
    s_fn(y,x) = collect(((y[n.+1].-y[n])./(x[n.+1].-y[n]).-(y[n].-y[n.-1])./(x[n].-y[n.-1]) for n=2:(length(x)-1))) # S vector
    c_fn(x) = collect(((x[n].-x[n-1])./6 for n=3:(length(x)-1))) # c vector (lower and upper diagonal)
    d_fn(x) = collect(((x[n].-x[n-2])./3 for n=3:length(x))) # d vector (middle diagonal)
    A_cs = A_ls
    B_cs = -A_cs.+1
    C_cs = zeros(length(xaxis),n_grid)
    D_cs = zeros(length(xaxis),n_grid)
    Vtilde_cs = zeros(length(xaxis),n_grid)
    for j = 1:n_grid
        # Get quantities for tridiagonal system
        c = c_fn(xi[j])
        d = d_fn(xi[j])
        s = s_fn(yi[j],xi[j])
        # Define tridiagonal matrix
        C = Tridiagonal(c,d,c)
        # Find second vector derivatives
        y_pp = [0;C\s;0]
        # Find C(x) and D(x) using A(x) and B(x) defined in the linear spline section
        C_cs[:,j] = (1/6).*((A_cs[:,j].^3).-A_cs[:,j]).*((xi[j][ind[:,j].+1].-xi[j][ind[:,j]]).^2)
        D_cs[:,j] = (1/6).*((B_cs[:,j].^3).-B_cs[:,j]).*((xi[j][ind[:,j].+1].-xi[j][ind[:,j]]).^2)
        # Interpolation function
        Vtilde_cs[:,j] = A_cs[:,j].*yi[j][ind[:,j]].+B_cs[:,j].*yi[j][ind[:,j].+1]+C_cs[:,j].*y_pp[ind[:,j]].+D_cs[:,j].*y_pp[ind[:,j].+1]
    end
    # Plot results
    gr()
    plt = plot(title="Interpolation n= (5,10,20) - Cubic spline")
    plot!(xaxis,V,linewidth=3,color=:black,label = "True uitility function",legend=(0.75,0.75),foreground_color_legend = nothing,background_color_legend = nothing)
    plot!(xaxis,Vtilde_cs[:,1],linestyle=:dash,linewidth=2,label="Interpolation - 5 points")
    plot!(xaxis,Vtilde_cs[:,2],linestyle=:dot,linewidth=2,label="Interpolation - 10 points")
    plot!(xaxis,Vtilde_cs[:,3],linestyle=:dashdot,linewidth=2,label="Interpolation - 20 points")
    savefig("./Figures/local_cubspline/U_fn$i.pdf")

    ## Assess interpolation accuracy with L2 norm (||V(x)-̃V(x)||_2) and print table
    acc = zeros(3,3)
    for j in 1:n_grid
        acc[1,j] = (sum((V.-Vtilde_m[:,j]).^2))^(1/2)
        acc[2,j] = (sum((V.-Vtilde_ls[:,j]).^2))^(1/2)
        acc[3,j] = (sum((V.-Vtilde_cs[:,j]).^2))^(1/2)
    end
    data = Any["Global monomial" map(v->round(v,digits=3),acc[1,:])'; "Linear spline" map(v->round(v,digits=3),acc[2,:])'; "cubic spline" map(v->round(v,digits=3),acc[3,:])']
    header = ["Interpolation method","5 grid points","10 grid points", "20 grid points"]
    pretty_table(data,header,backend = :latex)

    ## Plot interpolation accuracy as a function of grid size
    gr()
    plt = plot(title="Interpolation accuracy")
    plot!(n,acc[1,:],linewidth=2,marker=(:diamond,5),label="Global monomial")
    plot!(n,acc[2,:],linestyle=:dash,marker=(:diamond,5),linewidth=2,label="Linear spline")
    plot!(n,acc[3,:],linewidth=2,marker=(:diamond,5),label="Cubic spline")
    savefig("./Figures/accuracy_Ufn$i.pdf")
end

                        ### QUESTION_2 ###

# CRRA Utility function with σ=5
u(x) = (x.^(1-5))./(1-5)
# Function that does interpolation using linear spline and returns interpolated function in range xaxis
function linspline(yi,grid,xaxis)
    xi = collect(grid)
    n_range = length(xaxis)
    ind_fn(xi,x) = findmax(sign.(xi.-x))[2]-1
    ind = Array{Int64}(undef,n_range,1)
    A_ls = zeros(n_range,1)
    Vtilde_ls = zeros(n_range,1)
    # Find indices for local interpolation
    ind = map(x->ind_fn(xi,x),xaxis)
    # Find local weights (A(x) and B(x)=1-A(x))
    A_ls = (xi[ind.+1].-xaxis)./(xi[ind.+1].-xi[ind])
    # Find ̃V(x)
    Vtilde_ls = A_ls.*yi[ind].+ (-A_ls.+1).*yi[ind.+1]
    return Vtilde_ls
end
# Call routine that finds optimal grid curvature for CRRA utility with σ=5 using linear spline for interpolation
a = 0.05
b = 2
N = 10
opt_curv = optimal_gridcurv(a,b,N,u,linspline)
# Plot
θ_opt = round(opt_curv[1],digits=3)
Vtilde_opt = opt_curv[5]
V_true = u(xaxis)
gr()
plt = plot(title="Interpolation with optimal grid curvature θ=$θ_opt, N=10")
plot!(xaxis,Vtilde_opt,linewidth=2,label="Linear spline")
plot!(xaxis,V_true,linewidth=2,label="True function")
plot!(opt_curv[6],opt_curv[7],linetype=:scatter,marker=(:diamond,9),markercolor=RGB(0.5,0.1,0.1),label = "Data")
savefig("./Figures/Question_2/optimal_grid.pdf")
# Show results in table
data = Any[opt_curv[1] opt_curv[2] opt_curv[3] opt_curv[4]]
header = ["Optimal grid curvature θ","Sup norm of interpolated function vs true","Absolute differences in estimated θ", "Number of iterations to convergence"]
pretty_table(data,header,backend = :latex)

                        ### QUESTION_3 ###

# Define x-axis all the way to 2.5 for extrapolation
xaxis_extrapol = range(2,2.5,length=250)
xaxis_all = range(0.02,2.5,length=1250)
# Interpolation routine
for i in 1:n_f
    # y-grid is such that V(x) = ̃V(x) ∀ x=xi
    yi = U_fn(xi[2],i,σ)
    # True utility function V(x) - will be used as benchmark for all interpolation methods
    V_all = U_fn(collect(xaxis_all),i,σ)

                ### GLOBAL POLYNOMIAL INTERPOL WITH MONOMIAL BASIS (M) ###
    # Get polynomial coefficients
    a = mono_global(xi[2],yi)
    # ̃V(x) with monomial polynomial coefficients
    Vtilde(x,a) = sum(a[m]*(x.^(m-1)) for m=1:length(a))
    Vtilde_m = zeros(length(xaxis),n_grid) #
    Vtilde_m = Vtilde(xaxis,a) # Interpolated function within boundaries
    Vtilde_m_extrapol = Vtilde(xaxis_extrapol,a) # extrapolation of interpolated function
    # Plot the results
    gr()
    plt = plot(title="Interpolation n= 10 - Monomial Polynomial")
    plot!(xaxis_all,V_all,linewidth=2,color=:black,label = "True uitility function",foreground_color_legend = nothing,background_color_legend = nothing)
    plot!(xaxis,Vtilde_m,linestyle=:dash,linewidth=2,label="Interpolation - 10 points")
    plot!(xaxis_extrapol,Vtilde_m_extrapol,linestyle=:dash,linewidth=2,label="Extrapolation")
    savefig("./Figures/Extrapolation/U_fn$i.pdf")
end
