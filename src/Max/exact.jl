func = identity
multiple1 = 1.0
exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
x = ones(length(solver.vars))
exact_svd_bc = exact_svd_block(x)
multiple2 = 1/maximum(abs, exact_svd_bc)

ps = 1.0:0.5:6.0
hs = 0.0:4.0:20.0

projection = HeavisideProjection(0.0)
penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
obj = Objective(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection))
constr = BlockConstraint(multiple2 * func(exact_svd_block), multiple2 * func(70000.0))

lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-4, g_tol=1e-4, outer_iterations=10));

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
    p = ps[i]
    h = hs[1]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end

hs = 0.0:4:20.0
steps = length(hs)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 2:length(hs)
    p = ps[end]
    h = hs[i]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end


#=
for i in 1:2*length(ps)-1
    if i == 1
        p = ps[i]
        h = hs[i]
    elseif isodd(i)
        p = ps[(i + 1) ÷ 2]
        h = hs[(i + 1) ÷ 2]
    elseif iseven(i)
        p = ps[i ÷ 2]
        h = hs[(i ÷ 2) + 1]
    end
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
    if i == 1
        global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
    else
        _im.set_data(image)
    end
end

for i in 1:length(ps)
    p = ps[i]
    h = hs[i]
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=1e-5, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
end
=#

fname = "$out_dir/exact_svd_max_csimp"
if TopOpt.TopOptProblems.getdim(problem) == 2
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$fname.png")
    close()
end
save(
    "$fname.jld2",
    Dict("problem" => problem, "result" => result),
)
println("...............")
save_mesh(fname, problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))

@show maximum(exact_svd_block(result.minimizer))
@show obj(result.minimizer)
