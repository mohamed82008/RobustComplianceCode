Random.seed!(1)
println("Mean")

F = problem.F
V_rad = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hutch_rand!(V_rad)

V_had = zeros(eltype(F), size(F, 2), 1000)
TopOpt.hadamard!(V_had)

approx_means_rad = []
approx_means_had = []

function test_nvs(nvs = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    global approx_means_rad, approx_means_had
    x0 = fill(1.0, nels);
    exact = MeanCompliance(problem, solver; kwargs..., method = :exact)
    val, t, _, _, _ = @timed exact(x0)
    global exact_mean_val = val
    println("Exact: $val, Time: $t")
    println()

    exact_svd = MeanCompliance(problem, solver; kwargs..., method = :exact_svd)
    val, t, _, _, _ = @timed exact_svd(x0)
    println("Exact SVD: $val, Time: $t")
    println()

    println("Rademacher basis")
    for nv in nvs
        trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hutch, sample_once = true, nv = nv, V = V_rad)
        val, t, _, _, _ = @timed trace(x0)
        push!(approx_means_rad, val)
        println("nv = $nv, Val = $val, Time: $t")
    end
    println()

    println("Hadamard basis")
    for nv in nvs
        trace = MeanCompliance(problem, solver; kwargs..., method = :approx, sample_method = :hadamard, sample_once = true, nv = nv, V = V_had)
        val, t, _, _, _ = @timed trace(x0)
        push!(approx_means_had, val)
        println("nv = $nv, Val = $val, Time: $t")
    end
    println()
end
test_nvs()

plot([10; 100:100:1000], approx_means_rad, color = "black", linestyle="-", label="Rademacher estimate")
plot([10; 100:100:1000], approx_means_had, color = "black", linestyle="--", label="Hadamard estimate")
axhline(exact_mean_val, linestyle=":", color="grey", label="Exact")
legend()
title("Exact and approximate mean compliance")
savefig("$out_dir/exact_approx_mean.png")
close()