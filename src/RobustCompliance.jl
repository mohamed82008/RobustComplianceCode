module RobustCompliance

include("setup.jl")

@time include("Mean/exact.jl")
alert("Press any key to continue...")
readline()
@time include("Mean/trace.jl")
alert("Press any key to continue...")
readline()
@time include("MeanStd/exact.jl")
alert("Press any key to continue...")
readline()
@time include("MeanStd/diagonal_scaled_hadamard.jl")
alert("Press any key to continue...")
readline()

@time include("Max/exact.jl")
alert("Press any key to continue...")
readline()
@time include("Max/diagonal.jl")

include("Mean/probing_vectors.jl")
alert("Press any key to continue...")
readline()

include("Mean/correcting_factor.jl")
alert("Press any key to continue...")
readline()

include("MeanStd/probing_vectors.jl")
alert("Press any key to continue...")
readline()
include("MeanStd/correcting_factor.jl")
alert("Press any key to continue...")
readline()

alert("Your run is finished.")

end