exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
exact_svd_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), std)

x0 = fill(1.0, nels);
multiple = 1/exact_svd(x0)

ps = 1.0:0.5:6.0
maxtol = 1e-3
mintol = 1e-4
steps = length(ps)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 1:length(ps)
    p = ps[i]
    h = 0.0
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    obj = Objective(multiple * exact_svd)
    global volf = Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)
    constr = Constraint(volf, V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol =tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)    
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
    TopOpt.setpenalty!(solver, penalty)
    obj = Objective(multiple * exact_svd)
    global volf = Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)
    constr = Constraint(volf, V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol =tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)    
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
    TopOpt.setpenalty!(solver, penalty)
    obj = Objective(multiple * exact_svd)
    constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
    mma_options = MMA.Options(maxiter = 1000, s_incr = 1.05, tol = MMA.Tolerances(kkttol =tol))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)    
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

for i in 1:length(ps)
    p = ps[i]
    h = hs[i]
    println("=======")
    @show p, h
    println("=======")
    projection = HeavisideProjection(h)
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    obj = Objective(multiple * exact_svd)
    constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
    mma_options = MMA.Options(maxiter = 1000, s_init=0.01, s_incr = 1.05, tol = MMA.Tolerances(kkttol = 1e-4))
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)    
    simp = SIMP(optimizer, penalty.p)
    if i == 1
        global result = simp(x0)
    else
        global result = simp(result.topology)
    end
end
=#

fname = "$out_dir/exact_svd_mean_csimp"
if TopOpt.TopOptProblems.getdim(problem) == 2
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$fname.png")
    PyPlot.close()
end
save(
    "$fname.jld2",
    Dict("problem" => problem, "result" => result),
)
println("...............")
save_mesh(fname, problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))

@show exact_svd(result.topology)
@show exact_svd_std(result.topology)
bc = exact_svd_std.block(result.topology);
@show maximum(bc), minimum(bc)
@show volf(result.topology)
