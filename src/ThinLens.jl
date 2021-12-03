module ThinLens

import ChainRulesCore

greet() = print("Hello World!")

show() = print(@doc ChainRulesCore.rrule)

include("SplitSchemes.jl")
include("Transformations.jl")
include("Elements.jl")

end # module
