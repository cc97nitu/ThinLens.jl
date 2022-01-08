"""
This module implements existing beamlines in ThinLens.jl.
"""
module Lattice

import Flux
import ThinLens

function SIS18_Cell_minimal(k1f::Float64, k1d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd)
    bendingAngle = 0.2617993878
    rb1 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0)
    rb2 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0)

    d1 = ThinLens.Drift(0.645)
    d2 = ThinLens.Drift(0.9700000000000002)
    d3 = ThinLens.Drift(6.839011704000001)
    d4 = ThinLens.Drift(0.5999999999999979)
    d5 = ThinLens.Drift(0.7097999999999978)
    d6 = ThinLens.Drift(0.49979999100000283)

    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split)


    qs3t = ThinLens.Drift(0.4804)

    # set up beam line
    Flux.Chain(d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6,)
end


function SIS18_Lattice_minimal(k1f::Float64, k1d::Float64; nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd)
    if nested
        if cellsIdentical
            cell = SIS18_Cell_minimal(k1f, k1d; split=split)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell_minimal(k1f, k1d; split=split) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell_minimal(k1f, k1d; split=split)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell_minimal(k1f, k1d; split=split) for _ in 1:12])
        end
    end
end


function SIS18_Cell(k1f::Float64, k1d::Float64, k2f::Float64, k2d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd)
    # bending magnets
    bendingAngle = 0.2617993878
    rb1 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0)
    rb2 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0)

    # quadrupoles
    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split)
    qs3t = ThinLens.Drift(0.4804)

    # sextupoles
    ks1c = ThinLens.Sextupole(0.32, k2f, 0; split=split)
    ks3c = ThinLens.Sextupole(0.32, k2d, 0; split=split)
   
    # drifts
    d1 = ThinLens.Drift(0.66355)
    d2 = ThinLens.Drift(0.9700000000000002)
    d3a = ThinLens.Drift(6.345)
    d3b = ThinLens.Drift(0.175)
    d4 = ThinLens.Drift(0.5999999999999979)
    d5a = ThinLens.Drift(0.195)
    d5b = ThinLens.Drift(0.195)

    hMon = ThinLens.Drift(0.48125)

    # set up beamline
    #Flux.Chain(d1, rb1, d2, rb2 d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, d6a, hMon, vMonDrift, d6b)  # actual layout
    Flux.Chain(d1, rb1, d2, rb2, d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, hMon)
end


function SIS18_Lattice(k1f::Float64, k1d::Float64, k2f::Float64, k2d::Float64;
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd)

    if nested
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell(k1f, k1d, k2f, k2d; split=split) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell(k1f, k1d, k2f, k2d; split=split) for _ in 1:12])
        end
    end
end

end  # module