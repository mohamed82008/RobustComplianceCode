func = identity
multiple1 = 1.0
exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hadamard, sample_once = true, nv = 10)
Cmax = 70000.0

projection = HeavisideProjection(0.0)
penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
obj = Objective(func(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)))

## Continuation SIMP

x = ones(length(solver.vars))
diagonal_bc = diagonal_block(x)
exact_bc = exact_svd_block(x)
factor = Diagonal(exact_bc ./ diagonal_bc)
#factor = maximum(exact_bc) / maximum(diagonal_bc)
multiple2 = 1/(1000 * maximum(abs, exact_bc))

constr = BlockConstraint(multiple2 * func(factor * diagonal_block), multiple2 * func(Cmax))

lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-5, g_tol=1e-4, outer_iterations=10));

alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x));
AugLag.reset!(alg, λ = 1, r = 0.1, x = x);
w = 0.1; gamma = 3.0; dual_alpha0 = 1.0; primal_alpha0 = 1.0;
adapt_primal_step = 2; adapt_dual_step = 2
primal_cg = false; primal_step_adapt_factor0 = 1.5
dual_step_adapt_factor0 = 2.0
adapt_trust_region = false
primal_optim = TopOpt.AugLag.CG()
#primal_optim = TopOpt.AugLag.LBFGS(3)

ps = 1.0:0.5:6.0
maxtol = 1e-3
mintol = 1e-4
steps = length(ps)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 1:length(ps)
    global x, projection, diagonal_bc, factor, multiple2, constr, lag, optimizer, alg, result, _im
    @show maximum(diagonal_block(x))
    @show maximum(exact_svd_block(x))
    if false#i == 1
        diagonal_bc = diagonal_block(x)
        exact_bc = exact_svd_block(x)
        factor = Diagonal(exact_bc ./ diagonal_bc)
        multiple2 = 1/(maximum(exact_bc))

        constr = BlockConstraint(multiple2 * func(factor * diagonal_block), multiple2 * func(Cmax))
        optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-5, g_tol=1e-4, outer_iterations=10));
        lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
        alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x));
    end

    p = ps[i]
    h = 0.0
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    AugLag.reset!(alg, r = 0.1, x = x);
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
    if i == 1
        _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
    else
        _im.set_data(image)
    end
end

# Projection

hs = 0.0:3:15.0
maxtol = 1e-3
mintol = 1e-4
steps = length(hs)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 2:length(hs)
    global x, projection, diagonal_bc, factor, multiple2, constr, lag, optimizer, alg, result, _im
    @show maximum(diagonal_block(x))
    @show maximum(exact_svd_block(x))
    if false#i == 2
        diagonal_bc = diagonal_block(x)
        exact_bc = exact_svd_block(x)
        factor = Diagonal(exact_bc ./ diagonal_bc)
        multiple2 = 1/(maximum(exact_bc))

        constr = BlockConstraint(multiple2 * func(factor * diagonal_block), multiple2 * func(Cmax))
        lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
        optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-5, g_tol=1e-4, outer_iterations=10));
        alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x));
    end

    p = ps[end]
    h = hs[i]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
    if i == 1
        _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
    else
        _im.set_data(image)
    end
end

#=
ps = 1.0:1.0:5.0
hs = 0.0:4.0:16.0
x = similar(solver.vars); x .= 1.0;
primal_optim = TopOpt.AugLag.CG()

projection = HeavisideProjection(0.0)
penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
obj = Objective(func(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)))
constr = BlockConstraint(multiple2 * func(factor * diagonal_block), multiple2 * func(Cmax))

lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-5, g_tol=1e-4, outer_iterations=10));

alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x));
AugLag.reset!(alg, λ = 1, r = 0.1, x = x);
w = 0.1; gamma = 3.0; dual_alpha0 = 1.0; primal_alpha0 = 1.0;
adapt_primal_step = 2; adapt_dual_step = 2
primal_cg = false; primal_step_adapt_factor0 = 2.0
dual_step_adapt_factor0 = 2.0
adapt_trust_region = false
primal_optim = TopOpt.AugLag.CG()

for i in 1:length(ps)
    global x, projection
    p = ps[i]
    h = hs[i]
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=1e-5, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
end
=#

image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
PyPlot.imshow(image, cmap="Greys", origin="lower")
savefig("$out_dir/diagonal_max_csimp.png")
close()
save("$out_dir/diagonal_max_csimp.jld2", Dict("topology" => x))

@show maximum(factor * diagonal_block(result.minimizer))
@show maximum(exact_svd_block(result.minimizer))
@show obj(result.minimizer)
