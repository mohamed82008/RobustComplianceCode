# Super-secret repo that only reviewers are allowed to use :)

If you are a reviewer who would like to re-create the results, please use the following:

1. Clone or download this repository
2. Download and install Julia using https://julialang.org/downloads/
3. `cd` into this repository and run a Julia session.
4. Run the following Julia code:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
5. Change the code in `setup.jl` changing `out_root` to be the directory where you want to save the results.
6. Comment out the lines in `src/RobustCompliance.jl` that you want to avoid running.
7. Run `include("src/RobustCompliance.jl")` to run all the main experiments.
8. There are also additional experiments in `Max/additional` and `Mean/additional` that you have run separatelr if you are feeling adventerous.

Note: this code uses an old branch of TopOpt.jl that is preserved to maintain the validity of the code and consistency of the results. So the code here is not compatible with the master branch of TopOpt.jl.
