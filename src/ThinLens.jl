module ThinLens

import ChainRulesCore

greet() = print("Hello World!")

show() = print(@doc ChainRulesCore.rrule)

# core functionality
include("Beams.jl")
include("SplitSchemes.jl")
include("Transformations.jl")
include("Elements.jl")
include("Beamlines.jl")

# convenience methods
include("Lattice.jl")
include("Tools.jl")

Flux.@functor Drift
Flux.@functor Quadrupole
Flux.@functor Sextupole
Flux.@functor BendingMagnet
# Flux.@functor RBen
# Flux.@functor SBen



end # module
