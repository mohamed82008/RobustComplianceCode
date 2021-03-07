using Revise, TopOpt, Distributions, LinearAlgebra, Random, PyPlot, FileIO, JLD2
using TopOpt.AugLag: AugLag, IneqConstraintBlock, EqConstraintBlock, AugmentedPenalty, Lagrangian, AugmentedLagrangianAlgorithm, LinQuadAggregation
using TopOpt.Algorithms: BoxOptimizer

# Setup
    Random.seed!(1)
    E = 1.0; v = 0.3; xmin = 0.001;
    filterT = DensityFilter
    rmin = 2.0; V = 0.3

    f1 = RandomMagnitude([0, -1], Uniform(0.5, 1.5))
    f2 = RandomMagnitude(normalize([1, -1]), Uniform(0.5, 1.5))
    f3 = RandomMagnitude(normalize([-1, -1]), Uniform(0.5, 1.5))

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    kwargs = (filterT = filterT, tracing = true, logarithm = false, rmin = rmin)
    base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v)
    problem = MultiLoad(base_problem, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3], 1000)
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)

# Mean exact SVD
    exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
    exact_svd_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), std)

    x0 = fill(1.0, 160*40);
    multiple = 1/exact_svd(x0)
    ps = 1.0:1.0:5.0
    hs = 0.0:4.0:16.0
    #hs = [0.0 for i in 1:5]
    for i in 1:length(ps)
        p = ps[i]
        h = hs[i]
        projection = HeavisideProjection(h)
        penalty = ProjectedPenalty(PowerPenalty(p), projection);
        TopOpt.setpenalty!(solver, penalty)
        obj = Objective(multiple * exact_svd)
        constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
        mma_options = MMA.Options(maxiter = 1000, tol = MMA.Tolerances(kkttol = 1e-4))
        convcriteria = MMA.IpoptCriteria()
        optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)    
        simp = SIMP(optimizer, penalty.p)
        if i == 1
            global result = simp(x0)
        else
            global result = simp(result.topology)
        end
    end

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/exact_svd_mean_csimp.png")
    close()

    exact_svd(result.topology)
    exact_svd_std(result.topology)
    bc = exact_svd_std.block(result.topology);
    maximum(bc), minimum(bc)

# Mean trace
    trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hutch, sample_once = true, nv = 10)

    x0 = fill(1.0, 160*40);
    multiple = 1/trace(x0)
    ps = 1.0:1.0:5.0
    hs = 0.0:4.0:16.0
    for i in 1:length(ps)
        p = ps[i]
        h = hs[i]
        projection = HeavisideProjection(h)
        penalty = ProjectedPenalty(PowerPenalty(p), projection);
        TopOpt.setpenalty!(solver, penalty)
        obj = Objective(multiple * trace)
        constr = Constraint(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection), V)
        mma_options = MMA.Options(maxiter = 1000, tol = MMA.Tolerances(kkttol = 1e-4))
        convcriteria = MMA.IpoptCriteria()
        optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)
        simp = SIMP(optimizer, penalty.p)
        if i == 1
            global result = simp(x0)
        else
            global result = simp(result.topology)
        end
    end

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/trace_Rademacher10_mean_csimp.png")
    close()

    exact_svd(result.topology), trace(result.topology)
    exact_svd_std(result.topology)
    bc = exact_svd_std.block(result.topology);
    maximum(bc), minimum(bc)

# Mean-std
    exact_svd_mean_std = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), 0.5)

    x0 = fill(1.0, 160*40);
    multiple = 1/exact_svd_mean_std(x0)
    ps = 1.0:1.0:5.0
    hs = 0.0:4.0:16.0
    for i in 1:length(ps)
        p = ps[i]
        h = hs[i]
        projection = HeavisideProjection(h)
        penalty = ProjectedPenalty(PowerPenalty(p), projection);
        obj = Objective(multiple * exact_svd_mean_std)
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

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/exact_svd_mean_std_csimp.png")
    close()

    exact_svd(result.topology)
    exact_svd_std(result.topology)
    exact_svd_mean_std(result.topology)
    bc = exact_svd_std.block(result.topology);
    maximum(bc), minimum(bc)

# Mean-std diagonal scaled Hadamard
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
    diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = 10);
    x0 = fill(1.0, 160*40);
    exact_svd_bc = exact_svd_block(x0);
    diagonal_bc = diagonal_block(x0);
    factor1 = mean(exact_svd_bc) / mean(diagonal_bc)
    factor2 = std(exact_svd_bc) / std(diagonal_bc)
    diagonal_mean_std = ScalarValued(diagonal_block, (x) -> (factor1 * mean(x) + factor2 * 0.5 * std(x)))
    multiple = 1/(mean(exact_svd_bc) + 0.5 * std(exact_svd_bc))

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

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/diagonal_mean_std_csimp.png")
    close()

    exact_svd(result.topology)
    exact_svd_std(result.topology)
    exact_svd_mean_std(result.topology)
    bc = exact_svd_std.block(result.topology);
    maximum(bc), minimum(bc)

    diagonal_bc = diagonal_block(result.topology);
    factor1 * mean(diagonal_bc), factor2 * std(diagonal_bc)
    diagonal_mean_std(result.topology)
    maximum(diagonal_bc), minimum(diagonal_bc)

# Maximum compliance constraint - exact SVD
    func = identity
    multiple1 = 1.0
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
    x = ones(length(solver.vars))
    exact_svd_bc = exact_svd_block(x)
    multiple2 = 1/maximum(abs, exact_svd_bc)

    ps = 1.0:1.0:5.0
    hs = 0.0:4.0:16.0

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    obj = Objective(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection))
    constr = BlockConstraint(multiple2 * func(exact_svd_block), multiple2 * func(5000.0))

    lag = Lagrangian(AugmentedPenalty, obj, ineq=(constr,), r0 = 1.0, λ0 = 1.0)
    optimizer = BoxOptimizer(lag, Optim.ConjugateGradient(linesearch=HagerZhang(linesearchmax = 10)), options=Optim.Options(allow_outer_f_increases=false, allow_f_increases=false, x_tol=1e-5, f_tol=1e-4, g_tol=1e-4, outer_iterations=10));

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

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/exact_svd_max_csimp.png")
    close()
    save("$out_dir/exact_svd_max_csimp.jld2", Dict("topology" => x))

    maximum(exact_svd_block(result.minimizer))
    obj(result.minimizer)

# Maximum compliance constraint - diagonal
    func = identity
    multiple1 = 1.0
    diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hadamard, sample_once = true, nv = 10)
    x = ones(length(solver.vars))
    diagonal_bc = diagonal_block(x)
    factor = maximum(exact_svd_block(x)) / maximum(diagonal_block(x))
    multiple2 = 1/maximum(abs, diagonal_bc)

    ps = 1.0:1.0:5.0
    hs = 0.0:4.0:16.0
    x = similar(solver.vars); x .= 1.0;
    primal_optim = TopOpt.AugLag.CG()

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    obj = Objective(func(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection)))
    constr = BlockConstraint(multiple2 * func(factor * diagonal_block), multiple2 * func(5000.0))

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

    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/diagonal_max_csimp.png")
    close()
    save("$out_dir/diagonal_max_csimp.jld2", Dict("topology" => x))

    diagonal_bc = maximum(diagonal_block(result.minimizer)) * factor
    maximum(exact_svd_block(result.minimizer))
    obj(result.minimizer)

# Different probing vectors
    Random.seed!(1)
    function test_nvs(nvs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        x0 = fill(1.0, 160*40);
        exact = MeanCompliance(problem, solver; kwargs..., method = :exact)
        val, t, _, _, _ = @timed exact(x0)
        println("Exact: $val, Time: $t")
        println()

        exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
        val, t, _, _, _ = @timed exact_svd(x0)
        println("Exact SVD: $val, Time: $t")
        println()

        println("Rademacher basis")
        for nv in nvs
            trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hutch, sample_once = true, nv = nv)
            val, t, _, _, _ = @timed trace(x0)
            println("nv = $nv, Val = $val, Time: $t")
        end
        println()

        println("Hadamard basis")
        for nv in nvs
            trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hadamard, sample_once = true, nv = nv)
            val, t, _, _, _ = @timed trace(x0)
            println("nv = $nv, Val = $val, Time: $t")
        end
        println()
    end
    test_nvs()
# Correcting factor
    Random.seed!(1)
    function test_correction()
        nv = 10
        x0 = fill(1.0, 160*40);
        exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
        exact_val = exact_svd(x0)

        println("Rademacher basis")
        trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hutch, sample_once = true, nv = nv)
        approx_val = trace(x0)
        println("nv = $nv, Approx val = $approx_val, Ratio = $(exact_val/approx_val)")
        println()

        println("Hadamard basis")
        trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hadamard, sample_once = true, nv = nv)
        approx_val = trace(x0)
        println("nv = $nv, Approx val = $approx_val, Ratio = $(exact_val/approx_val)")
        println()
    end
    test_correction()

    function plot_histo(run=true)
        Random.seed!(1)
        nv = 10
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]
            ms = "0"*string(Int(m*10))
            if run
                dist = Truncated(Normal(m, 0.2), 0, 1)
                exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
                trace_rad = MeanCompliance(problem, solver; kwargs..., method = :approx, 
                    sample_method = :hutch, sample_once = true, nv = nv)
                trace_had = MeanCompliance(problem, solver; kwargs..., method = :approx, 
                    sample_method = :hadamard, sample_once = true, nv = nv)
                factors_rad = Float64[]
                factors_had = Float64[]
                for i in 1:500
                    x0 = rand(dist, 160*40);
                    exact_val = exact_svd(x0)
                    approx_val_rad = trace_rad(x0)
                    push!(factors_rad, exact_val/approx_val_rad)
                    approx_val_had = trace_had(x0)
                    push!(factors_had, exact_val/approx_val_had)
                end
            else
                factors_rad = load("correcting_factors_rademacher_$(ms).jld2")["factors"]
                factors_had = load("correcting_factors_hadamard.jld2")["factors"]
            end

            hist(factors_rad, color = :black)
            xlabel("Correcting factor")
            ylabel("Count")
            savefig("$out_dir/correcting_factors_rademacher_mean_$(nv)_$(ms).png")
            close()

            hist(factors_had, color = :black)
            xlabel("Correcting factor")
            ylabel("Count")
            savefig("$out_dir/correcting_factors_hadamard_mean_$(nv)_$(ms).png")
            close()

            #histogram(factors_rad, ylabel = "Count", xlabel = "Correcting factor")
            #savefig("$out_dir/correcting_factors_rademacher.png")
            #closeall()
            if run
                save("$out_dir/correcting_factors_rademacher_$(nv)_$(ms).jld2", Dict("factors" => factors_rad))
                save("$out_dir/correcting_factors_hadamard_$(nv)_$(ms).jld2", Dict("factors" => factors_had))
            end
        end
    end
    plot_histo(false)

# Different probing vectors - mean-std
    function test_nvs2(nvs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        Random.seed!(1)
        x0 = fill(1.0, 160*40);
        exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
        exact_svd_bc = exact_svd_block(x0);
        println("Exact-SVD")
        mu = mean(exact_svd_bc)
        _std = std(exact_svd_bc)
        println("mu = $mu, std = $_std, sum = $(mu + 0.5_std)")

        println("Rademacher basis")
        for nv in nvs
            diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hutch, sample_once = true, nv = nv);
            diagonal_bc = diagonal_block(x0);
            mu = mean(diagonal_bc)
            _std = std(diagonal_bc)
            println("nv = $nv, mu = $mu, std = $_std, sum = $(mu + 0.5 * _std)")
        end
        println()

        println("Hadamard basis")
        for nv in nvs
            diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv);
            diagonal_bc = diagonal_block(x0);
            mu = mean(diagonal_bc)
            _std = std(diagonal_bc)
            println("nv = $nv, mu = $mu, std = $_std, sum = $(mu + 0.5 * _std)")
        end
        println()
    end
    test_nvs2()

# Correcting factor - std
    function test_correction()
        Random.seed!(1)
        nv = 10
        x0 = fill(1.0, 160*40);
        exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
        exact_svd_bc = exact_svd_block(x0);
        exact_std_val = std(exact_svd_bc)
        diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv);
        diagonal_bc = diagonal_block(x0);
        approx_std_val = std(diagonal_bc)
        
        println("Hadamard basis")
        println("nv = $nv, Approx val = $approx_std_val, Ratio = $(exact_std_val/approx_std_val)")
        println()
    end
    test_correction()

    function plot_histo2(run=true)
        Random.seed!(1)
        nv = 10
        if run
            exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
            diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv);
        end
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]
            ms = "0"*string(Int(m*10))
            if run
                dist = Truncated(Normal(m, 0.2), 0, 1)
                factors_had = Float64[]
                for i in 1:500
                    x0 = rand(dist, 160*40);
                    exact_svd_bc = exact_svd_block(x0);
                    exact_std_val = std(exact_svd_bc)
                    diagonal_bc = diagonal_block(x0);
                    approx_std_val = std(diagonal_bc)
                    push!(factors_had, exact_std_val/approx_std_val)
                end
            else
                factors_had = load("$out_dir/correcting_factors_hadamard_$(nv)_$(ms).jld2")["factors"]
            end
            fig, ax = plt.subplots()
            ax.hist(factors_had, color = :black)
            xlabel("Correcting factor")
            ylabel("Count")
            ax.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%0.5f"))
            savefig("$out_dir/correcting_factors_hadamard_std_$(nv)_$(ms).png")
            close()
            if run
                save("$out_dir/correcting_factors_hadamard_std_$(nv)_$(ms).jld2", Dict("factors" => factors_had))
            end
        end
    end
    plot_histo2(false)


####

# MMA-AugLag - maximum compliance constraint - exact SVD
    func = identity
    multiple1 = 1.0
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
    x = ones(length(solver.vars))
    exact_svd_bc = exact_svd_block(x)
    #multiple2 = 1/maximum(abs, exact_svd_bc)
    multiple2 = 1.0

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    obj = Objective(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection))
    constr = BlockConstraint(multiple2 * func(exact_svd_block), multiple2 * func(5000.0))
    mma_options = MMA.Options(maxiter = 500, tol = MMA.Tolerances(kkttol = 1e-4), s_init = 0.1, s_incr = 1.1, s_decr = 0.9)
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMALag.MMALag20(MMA.MMA87(), true), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)

    x0 = fill(1.0, 160*40);
    ps = 1.0:1.0:6.0
    hs = 0.0:4.0:20.0
    i = 1
    for i in 1:length(ps)
        p = ps[i]
        h = hs[i]
        projection.β = h
        penalty = ProjectedPenalty(PowerPenalty(p), projection);
        TopOpt.setpenalty!(solver, penalty)
        simp = SIMP(optimizer, penalty.p)
        if i == 1
            global result = simp(x0)
        else
            global result = simp(result.topology)
        end
    end

    #=
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$out_dir/mma_auglag_exact_svd_max_csimp.png")
    close()
    save("$out_dir/mma_auglag_exact_svd_max_csimp.jld2", Dict("topology" => result.topology))
    =#
