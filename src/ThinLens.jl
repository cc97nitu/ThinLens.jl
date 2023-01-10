module ThinLens

import TaylorSeries
import TaylorSeries as TS

import ChainRulesCore
import Flux
using SnoopPrecompile


greet() = print("Hello World!")

show() = print(@doc ChainRulesCore.rrule)

# core functionality
include("Beams.jl")
include("SplitSchemes.jl")
include("Transformations.jl")
include("Elements.jl")
include("Beamlines.jl")

# thick tracking
include("ThickTrack.jl")
include("Polynomial.jl")

# convenience methods
include("Lattice.jl")
include("Tools.jl")

@precompile_setup begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    particle = [-8.16e-03, -1.78e-03,  5.55e-03,  1.93e-03,  0.00e+00,  7.87e-04, 9.99e-01];
    model = Lattice.SIS18_Lattice(0.3509739273586273, -0.3090477494318206, 0., 0.; split=splitO2nd, steps=4);
    accelerator = Lattice.SIS18_Lattice(0.3509739273586273, -0.3090477494318206, 0.01, 0.; split=splitO2nd, steps=4);
    label = accelerator(particle)

    @precompile_all_calls begin
        # all calls in this block will be precompiled, regardless of whether
        # they belong to your package or not (on Julia 1.8 and higher)
        sample = model(particle)
        parameters = Flux.params(model)
        grads = Flux.gradient(parameters) do
            Flux.Losses.mse(sample, label)
        end
    end
end


end # module
