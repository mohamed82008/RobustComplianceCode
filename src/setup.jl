using Revise, TopOpt, Distributions, LinearAlgebra, Random, PyPlot, FileIO, JLD2
using TopOpt.AugLag: AugLag, IneqConstraintBlock, EqConstraintBlock, AugmentedPenalty, Lagrangian, AugmentedLagrangianAlgorithm, LinQuadAggregation
using TopOpt.Algorithms: BoxOptimizer
using Alert, SparseArrays

# Setup

Random.seed!(1)
E = 1.0; v = 0.3; xmin = 0.001;
filterT = DensityFilter
dim = 2

out_root = "E:/Mohamed/RobustComplianceResults/"

@assert isdir(out_root) "Results directory not set properly."

if dim == 2
    out_dir = "$out_root/results_2d"
    rmin = 2.0; V = 0.4

    base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v)
    f1 = RandomMagnitude([0, -1], Uniform(-2.0, 2.0))
    f2 = RandomMagnitude(normalize([1, -1]), Uniform(-2.0, 2.0))
    f3 = RandomMagnitude(normalize([-1, -1]), Uniform(-2.0, 2.0))
    problem = MultiLoad(base_problem, 1000, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3])
    #problem = MultiLoad(base_problem, 1000, Uniform(-2, 2))
    nels = 160*40
else
    out_dir = "$out_root/results_3d"
    rmin = 3.0; V = 0.4

    base_problem = PointLoadCantilever(Val{:Linear}, (60, 20, 20), (1.0, 1.0, 1.0), E, v)
    f1 = RandomMagnitude([0, -1, 0], Uniform(-2.0, 2.0))
    f2 = RandomMagnitude(normalize([1, -1, 0]), Uniform(-2.0, 2.0))
    f3 = RandomMagnitude(normalize([-1, -1, 0]), Uniform(-2.0, 2.0))
    problem = MultiLoad(base_problem, 1000, [(60, 20, 10) => f1, (30, 20, 10) => f2, (40, 0, 10) => f3])
    #problem = MultiLoad(base_problem, 1000, Uniform(-2, 2))
    nels = 60*20*20
end

dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
dense_rank = 7
F = problem.F
for i in 1:dense_rank
    global F
    F += sparsevec(dense_load_inds, randn(length(dense_load_inds)) / dense_rank, size(problem.F, 1)) * randn(size(problem.F, 2))'
end
problem = MultiLoad(problem.problem, F)
rows = unique(F.rowval)
Fc = Matrix(F[rows, :])
svdfact = svd(Fc)
threshold = 1e-6
force_rank = length(findall(x -> x > threshold, svdfact.S))
@show force_rank

projection = HeavisideProjection(0.0)
penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
kwargs = (filterT = filterT, tracing = true, logarithm = false, rmin = rmin)
#if dim == 2
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)
#else
#    solver = FEASolver(Displacement, CG, Assembly, problem, xmin = xmin, penalty = penalty, cg_max_iter = 700)
#end
