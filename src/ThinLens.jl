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

end # module
