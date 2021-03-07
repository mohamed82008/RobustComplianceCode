exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
exact_svd_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), std)
exact_svd_mean_std = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), 2.0)

exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = 10);
x0 = fill(1.0, nels);
exact_svd_bc = exact_svd_block(x0);
diagonal_bc = diagonal_block(x0);
factor1 = mean(exact_svd_bc) / mean(diagonal_bc)
factor2 = std(exact_svd_bc) / std(diagonal_bc)
diagonal_mean_std = ScalarValued(diagonal_block, (x) -> (factor1 * mean(x) + factor2 * 2.0 * std(x)))
multiple = 1/(mean(exact_svd_bc) + 2.0 * std(exact_svd_bc))

ps = 1.0:0.5:6.0
maxtol = 1e-3
mintol = 1e-4
steps = length(ps)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 1:length(ps)
    h = 0.0
    p = ps[i]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    obj = Objective(multiple * diagonal_mean_std)
    global volf = Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)
    constr = Constraint(volf, V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol = tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)
        TopOpt.setpenalty!(solver, penalty)
    simp = SIMP(optimizer, penalty.p)
    if i == 1
        global result = simp(x0)
    else
        global result = simp(result.topology)
    end
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end

hs = 0.0:4.0:20.0
maxtol = 1e-3
mintol = 1e-4
steps = length(hs)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 2:length(hs)
    h = hs[i]
    p = ps[end]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    obj = Objective(multiple * diagonal_mean_std)
    global volf = Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)
    constr = Constraint(volf, V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol = tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)
        TopOpt.setpenalty!(solver, penalty)
    simp = SIMP(optimizer, penalty.p)
    if i == 1
        global result = simp(x0)
    else
        global result = simp(result.topology)
    end
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end

#=
ps = 1.0:0.1:6.0
hend = 10.0
hs = 0.0:(hend/(length(ps) - 1)):hend
maxtol = 1e-3
mintol = 1e-4
steps = 2*length(ps)-2
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

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
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    obj = Objective(multiple * diagonal_mean_std)
    constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol = tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)
        TopOpt.setpenalty!(solver, penalty)
    simp = SIMP(optimizer, penalty.p)
    if i == 1
        global result = simp(x0)
    else
        global result = simp(result.topology)
    end
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    if i == 1
        global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
    else
        _im.set_data(image)
    end
end

ps = 1.0:1.0:5.0
hs = 0.0:4.0:16.0
for i in 1:length(ps)
    p = ps[i]
    h = hs[i]
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    obj = Objective(multiple * diagonal_mean_std)
    constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
    mma_options = MMA.Options(maxiter = 1000, tol = MMA.Tolerances(kkttol = 1e-4))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)
        TopOpt.setpenalty!(solver, penalty)
    simp = SIMP(optimizer, penalty.p)
    if i == 1
        global result = simp(x0)
    else
        global result = simp(result.topology)
    end
end
=#

fname = "$out_dir/diagonal_mean_std_csimp"
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

@show exact_svd(result.topology)
@show exact_svd_std(result.topology)
@show exact_svd_mean_std(result.topology)
bc = exact_svd_std.block(result.topology);
@show maximum(bc), minimum(bc)

diagonal_bc = diagonal_block(result.topology);
@show factor1 * mean(diagonal_bc), factor2 * std(diagonal_bc)
@show diagonal_mean_std(result.topology)
@show maximum(diagonal_bc), minimum(diagonal_bc)
@show volf(result.topology)
