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

exact = MeanCompliance(problem, solver; kwargs..., method = :exact)
exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
trace1 = MeanCompliance(problem, solver; kwargs..., method = :trace, nv = 100, sample_once = true, sample_method = :hutch)
trace2 = MeanCompliance(problem, solver; kwargs..., method = :trace, nv = 100, sample_once = true, sample_method = :hadamard)

x = ones(length(solver.vars))
g = similar(x)

exact(x, g); @time exact(x, g);
exact_svd(x, g); @time exact_svd(x, g);
trace1(x, g); @time trace1(x, g);
trace2(x, g); @time trace2(x, g);
