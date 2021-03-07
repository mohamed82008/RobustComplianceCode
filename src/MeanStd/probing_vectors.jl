Random.seed!(1)
println("Std")

F = problem.F
V_rad = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hutch_rand!(V_rad)

V_had = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hadamard!(V_had)

approx_stds_rad = []
approx_stds_had = []

function test_nvs(nvs = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    global approx_stds_rad, approx_stds_had
    x0 = fill(1.0, nels);
    exact = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact), std)
    val, t, _, _, _ = @timed exact(x0)
    global exact_std_val = val
    println("Exact: $val, Time: $t")
    println()

    exact_svd = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :exact_svd), std)
    val, t, _, _, _ = @timed exact_svd(x0)
    println("Exact SVD: $val, Time: $t")
    println()

    println("Rademacher basis")
    for nv in nvs
        diag_est = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hutch, sample_once = true, nv = nv, V = V_rad), std)
        val, t, _, _, _ = @timed diag_est(x0)
        push!(approx_stds_rad, val)
        println("nv = $nv, Val = $val, Time: $t")
    end
    println()

    println("Hadamard basis")
    for nv in nvs
        trace = ScalarValued(BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv, V = V_had), std)
        val, t, _, _, _ = @timed trace(x0)
        push!(approx_stds_had, val)
        println("nv = $nv, Val = $val, Time: $t")
    end
    println()
end
test_nvs()

plot([10; 100:100:1000], approx_stds_rad, color = "black", linestyle="-", label="Rademacher estimate")
plot([10; 100:100:1000], approx_stds_had, color = "black", linestyle="--", label="Hadamard estimate")
axhline(exact_std_val, linestyle=":", color="grey", label="Exact")
legend()
title("Exact and approximate compliance standard deviation")
savefig("$out_dir/exact_approx_std.png")
close()

#=
Random.seed!(1)
println("MeanStd")

F = problem.F
V_rad = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hutch_rand!(V_rad)

V_had = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hadamard!(V_had)

approx_stds_rad = []
approx_stds_had = []

function test_nvs2(nvs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    global approx_stds_rad, approx_stds_had
    x0 = fill(1.0, nels);
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
    exact_svd_bc = exact_svd_block(x0);
    println("Exact-SVD")
    mu = mean(exact_svd_bc)
    _std = std(exact_svd_bc)
    global exact_std_val = _std
    println("mu = $mu, std = $_std, sum = $(mu + 0.5_std)")

    println("Rademacher basis")
    for nv in nvs
        diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hutch, sample_once = true, nv = nv, V = V_rad);
        diagonal_bc = diagonal_block(x0);
        mu = mean(diagonal_bc)
        _std = std(diagonal_bc)
        _std, t, _, _, _ = @timed trace(x0)
        push!(approx_stds_rad, _std)
        println("nv = $nv, mu = $mu, std = $_std, sum = $(mu + 0.5 * _std), time = $t")
    end
    println()

    println("Hadamard basis")
    for nv in nvs
        diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv, V = V_had);
        diagonal_bc = diagonal_block(x0);
        mu = mean(diagonal_bc)
        _std = std(diagonal_bc)
        push!(approx_stds_had, _std)
        println("nv = $nv, mu = $mu, std = $_std, sum = $(mu + 0.5 * _std)")
    end
    println()
end
test_nvs2()

plot(100:100:1000, approx_stds_rad, color = "black", linestyle="-", label="Rademacher estimate")
plot(100:100:1000, approx_stds_had, color = "black", linestyle="--", label="Hadamard estimate")
axhline(exact_std_val, linestyle=":", color="grey", label="Exact")
legend()
title("Exact and approximate compliance standard deviation")
savefig("$out_dir/exact_approx_std.png")
close()
=#