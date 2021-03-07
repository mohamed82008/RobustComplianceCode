function test_correction()
    Random.seed!(1)
    nv = 10
    x0 = fill(1.0, nels);
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd);
    exact_svd_bc = exact_svd_block(x0);
    exact_std_val = std(exact_svd_bc)
    diagonal_block = BlockCompliance(problem, solver; kwargs..., method = :diagonal, sample_method = :hadamard, sample_once = true, nv = nv);
    diagonal_bc = diagonal_block(x0);
    approx_std_val = std(diagonal_bc)
    ratios = (exact_svd_bc ./ diagonal_bc)
    _max, _min = maximum(ratios), minimum(ratios)

    println("Hadamard basis")
    println("nv = $nv, Approx val = $approx_std_val, Ratio = $(exact_std_val/approx_std_val), max_diag_ratio = $_max, min_diag_ratio = $_min")
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
            dist = Truncated(Normal(m, 0.1), 0, 1)
            factors_had = Float64[]
            for i in 1:500
                x0 = rand(dist, nels);
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
        ax.hist(factors_had, color = :black, bins=6)
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
plot_histo2(true)
