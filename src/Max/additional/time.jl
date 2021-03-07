using Revise, TopOpt, Distributions, LinearAlgebra, Random

Random.seed!(1)
E = 1.0; v = 0.3; xmin = 0.001;
filterT = DensityFilter
rmin = 2.0

f1 = RandomMagnitude([0, -1], Uniform(0.5, 1.5))
f2 = RandomMagnitude(normalize([1, -1]), Uniform(0.5, 1.5))
f3 = RandomMagnitude(normalize([-1, -1]), Uniform(0.5, 1.5))

p = 3.0
h = 10.0
projection = HeavisideProjection(h)
penalty = ProjectedPenalty(PowerPenalty(p), projection);
kwargs = (filterT = filterT, tracing = true, logarithm = false, rmin = rmin)
base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v)
problem = MultiLoad(base_problem, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3], 1000)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)

exact_mean = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact), mean)
exact_svd_mean = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), mean)
diagonal1_mean = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hadamard, nv = 100), mean)
diagonal2_mean = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hutch, nv = 100), mean)

exact_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact), std)
exact_svd_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), std)
diagonal1_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hadamard, nv = 100), std)
diagonal2_std = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hutch, nv = 100), std)

exact = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :exact), 0.5)
exact_svd = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), 0.5)
diagonal1 = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hadamard, nv = 100), 0.5)
diagonal2 = MeanStd(BlockCompliance(problem, solver; kwargs..., method = :approx, sample_once = true, sample_method = :hutch, nv = 100), 0.5)

x = ones(length(solver.vars))
g = similar(x)

exact_mean(x, g); @time exact_mean(x, g)
exact_std(x, g); @time exact_std(x, g)
exact(x, g); @time exact(x, g)

exact_svd_mean(x, g); @time exact_svd_mean(x, g)
exact_svd_std(x, g); @time exact_svd_std(x, g)
exact_svd(x, g); @time exact_svd(x, g)

diagonal1_mean(x, g); @time diagonal1_mean(x, g)
diagonal1_std(x, g); @time diagonal1_std(x, g)
diagonal1(x, g); @time diagonal1(x, g)

diagonal2_mean(x, g); @time diagonal2_mean(x, g)
diagonal2_std(x, g); @time diagonal2_std(x, g)
diagonal2(x, g); @time diagonal2(x, g)

diffs = map([rand(length(solver.vars)) for i in 1:100]) do x
    diagonal(x, g) - exact_svd(x, g)
end
