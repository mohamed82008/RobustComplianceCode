function test_correction()
    Random.seed!(1)
    nv = 10
    x0 = fill(1.0, nels);
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
                x0 = rand(dist, nels);
                exact_val = exact_svd(x0)
                #approx_val_rad = trace_rad(x0)
                #push!(factors_rad, exact_val/approx_val_rad)
                approx_val_had = trace_had(x0)
                push!(factors_had, exact_val/approx_val_had)
            end
        else
            #factors_rad = load("$out_dir/correcting_factors_rademacher_$(nv)_$(ms).jld2")["factors"]
            factors_had = load("$out_dir/correcting_factors_hadamard_$(nv)_$(ms).jld2")["factors"]
        end

        #hist(factors_rad, color = :black, bins=6)
        #xlabel("Correcting factor")
        #ylabel("Count")
        #savefig("$out_dir/correcting_factors_rademacher_mean_$(nv)_$(ms).png")
        #close()

        fig, ax = PyPlot.subplots()
        hist(factors_had, color = :black, bins=6)
        xlabel("Correcting factor")
        ylabel("Count")
        savefig("$out_dir/correcting_factors_hadamard_mean_$(nv)_$(ms).png")
        ax.yaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.3f"))
        close()

        #histogram(factors_rad, ylabel = "Count", xlabel = "Correcting factor")
        #savefig("$out_dir/correcting_factors_rademacher.png")
        #closeall()
        if run
            #save("$out_dir/correcting_factors_rademacher_$(nv)_$(ms).jld2", Dict("factors" => factors_rad))
            save("$out_dir/correcting_factors_hadamard_$(nv)_$(ms).jld2", Dict("factors" => factors_had))
        end
    end
end
plot_histo(true)
