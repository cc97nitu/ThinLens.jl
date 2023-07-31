"""
This module implements existing beamlines in ThinLens.jl.
"""
module Lattice

import Flux
import ThinLens


function FODO(k1f::Float64, k1d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    # d1 = ThinLens.Drift(1.)
    # d2 = ThinLens.Drift(1.)
    # d3 = ThinLens.Drift(1.)

    qs1f = ThinLens.Quadrupole(1., k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1., k1d, 0; split=split, steps=steps)

    # set up beam line
    # Flux.Chain(d1, qs1f, d2, qs2d, d3)
    Flux.Chain(qs1f, qs2d)

end


function SIS18_Cell_minimal(k1f::Float64, k1d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    bendingAngle = 0.2617993878
    rb1 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0; split=split, steps=steps)
    rb2 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0; split=split, steps=steps)

    d1 = ThinLens.Drift(0.645)
    d2 = ThinLens.Drift(0.9700000000000002)
    d3 = ThinLens.Drift(6.839011704000001)
    d4 = ThinLens.Drift(0.5999999999999979)
    d5 = ThinLens.Drift(0.7097999999999978)
    d6 = ThinLens.Drift(0.49979999100000283)

    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)


    qs3t = ThinLens.Drift(0.4804)

    # set up beam line
    Flux.Chain(d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6,)
end


function SIS18_Lattice_minimal(k1f::Float64, k1d::Float64; nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    if nested
        if cellsIdentical
            cell = SIS18_Cell_minimal(k1f, k1d; split=split, steps=steps)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell_minimal(k1f, k1d; split=split, steps=steps) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell_minimal(k1f, k1d; split=split, steps=steps)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell_minimal(k1f, k1d; split=split, steps=steps) for _ in 1:12])
        end
    end
end


function SIS18_Cell_FODO(k1f::Float64, k1d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    d1 = ThinLens.Drift(0.645 + 0.9700000000000002 + 2*2.617993878 + 6.839011704000001)
    d4 = ThinLens.Drift(0.5999999999999979)
    d5 = ThinLens.Drift(0.7097999999999978 + 0.4804 + 0.49979999100000283)

    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)

    # set up beam line
    Flux.Chain(d1, qs1f, d4, qs2d, d5)
end


function SIS18_Lattice_FODO(k1f::Float64, k1d::Float64, k2f::Float64, k2d::Float64;
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)

    if nested
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell_FODO(k1f, k1d; split=split, steps=steps) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell_FODO(k1f, k1d; split=split, steps=steps) for _ in 1:12])
        end
    end
end


function SIS18_Cell_noDipoles(k1f::Float64=0.364439095886585, k1d::Float64=-0.34369003221817673, k2f::Float64=0., k2d::Float64=0.; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    # bending magnets
    bendingAngle = 0.2617993878
    rb1 = ThinLens.Drift(2.617993878)
    rb2 = ThinLens.Drift(2.617993878)

    # quadrupoles
    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)
    qs3t = ThinLens.Drift(0.4804)

    # sextupoles
    ks1c = ThinLens.Sextupole(0.32, k2f, 0; split=split, steps=steps)
    ks3c = ThinLens.Sextupole(0.32, k2d, 0; split=split, steps=steps)
   
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
    Flux.Chain(d1, rb1, d2, rb2, d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, hMon)
end


function SIS18_Lattice_noDipoles(k1f::Float64=0.364439095886585, k1d::Float64=-0.34369003221817673, k2f::Float64=0., k2d::Float64=0.;
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)

    if nested
        if cellsIdentical
            cell = SIS18_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps) for _ in 1:12])
        end
    end
end


function SIS18_Cell(k1f::Float64, k1d::Float64, k2f::Float64, k2d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    # bending magnets
    bendingAngle = 0.2617993878
    rb1 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0; split=split, steps=steps)
    rb2 = ThinLens.RBen(2.617993878, bendingAngle, 0, 0; split=split, steps=steps)

    # quadrupoles
    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)
    qs3t = ThinLens.Drift(0.4804)

    # sextupoles
    ks1c = ThinLens.Sextupole(0.32, k2f, 0; split=split, steps=steps)
    ks3c = ThinLens.Sextupole(0.32, k2d, 0; split=split, steps=steps)
   
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
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)

    if nested
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.NestedChain([cell for _ in 1:12])
        else
            return ThinLens.NestedChain([SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps) for _ in 1:12])
        end
    else
        if cellsIdentical
            cell = SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps)
            return ThinLens.FlatChain([cell for _ in 1:12])
        else
            return ThinLens.FlatChain([SIS18_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps) for _ in 1:12])
        end
    end
end


# adapted to El Hayek Lattice via MAD-X twiss table
function SIS18_Cell_triplet_odd(k1f::Float64, k1d::Float64, k1t::Float64, k2f::Float64, k2d::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    # bending magnets
    # bendingAngle = 0.2725332
    bendingAngle = 0.2617993877991494
    rb1 = ThinLens.RBen(2.725332, bendingAngle, 0, 0, 0.045, 0.35; split=split, steps=steps)
    rb2 = ThinLens.RBen(2.725332, bendingAngle, 0, 0, 0.045, 0.35; split=split, steps=steps)

    # quadrupoles
    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)
    qs3t = ThinLens.Quadrupole(0.4804, k1t, 0; split=split, steps=steps)

    # sextupoles
    ks1c = ThinLens.Sextupole(0.32, k2f, 0; split=split, steps=steps)
    ks3c = ThinLens.Sextupole(0.32, k2d, 0; split=split, steps=steps)
   
    # drifts
    d1 = ThinLens.Drift(0.221334)
    d2 = ThinLens.Drift(0.862668)
    d3a = ThinLens.Drift(6.290334)
    d3b = ThinLens.Drift(0.175)
    d4 = ThinLens.Drift(0.6)
    d5a = ThinLens.Drift(0.195)
    d5b = ThinLens.Drift(0.195)

    hMon = ThinLens.Drift(0.87)

    # set up beamline
    #Flux.Chain(d1, rb1, d2, rb2 d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, d6a, hMon, vMonDrift, d6b)  # actual layout
    Flux.Chain(d1, rb1, d2, rb2, d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, hMon)
end


function SIS18_Cell_triplet_even(k1f::Float64, k1d::Float64, k1t::Float64; split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1)
    # bending magnets
    # bendingAngle = 0.2725332
    bendingAngle = 0.2617993877991494
    rb1 = ThinLens.RBen(2.725332, bendingAngle, 0, 0, 0.045, 0.35; split=split, steps=steps)
    rb2 = ThinLens.RBen(2.725332, bendingAngle, 0, 0, 0.045, 0.35; split=split, steps=steps)

    # quadrupoles
    qs1f = ThinLens.Quadrupole(1.04, k1f, 0; split=split, steps=steps)
    qs2d = ThinLens.Quadrupole(1.04, k1d, 0; split=split, steps=steps)
    qs3t = ThinLens.Quadrupole(0.4804, k1t, 0; split=split, steps=steps)
   
    # drifts
    d1 = ThinLens.Drift(0.221334)
    d2 = ThinLens.Drift(0.862668)
    d3 = ThinLens.Drift(6.290334 + 0.32 + 0.175)
    d4 = ThinLens.Drift(0.6)
    d5 = ThinLens.Drift(0.195 + 0.32 + 0.195)

    hMon = ThinLens.Drift(0.87)

    # set up beamline
    #Flux.Chain(d1, rb1, d2, rb2 d3a, ks1c, d3b, qs1f, d4, qs2d, d5a, ks3c, d5b, qs3t, d6a, hMon, vMonDrift, d6b)  # actual layout
    Flux.Chain(d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, hMon)
end


function SIS18_Lattice_QKicker(k1f::Float64, k1d::Float64, k1t::Float64, k2f::Float64, k2d::Float64;
    split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1, cellsIdentical::Bool=false)   
    # distance between kicker GS05MQ1 to GS05KS1C
    d1 = ThinLens.Drift(3.082)

    # set up beamline
    if cellsIdentical
        even_cell = SIS18_Cell_triplet_even(k1f, k1d, k1t; split=split, steps=steps)
        odd_cell = SIS18_Cell_triplet_odd(k1f, k1d, k1t, k2f, k2d; split=split, steps=steps)

        cells_even = [even_cell for _ in 1:6]
        cells_odd = [odd_cell for _ in 1:6]
    else
        cells_even = [SIS18_Cell_triplet_even(k1f, k1d, k1t; split=split, steps=steps) for _ in 1:6]
        cells_odd = [SIS18_Cell_triplet_odd(k1f, k1d, k1t, k2f, k2d; split=split, steps=steps) for _ in 1:6]

    end

    lattice = []
    for i in 1:6
        push!(lattice, cells_even[i])
        push!(lattice, cells_odd[i])
    end

    # add pretrack and after drift section
    GS05MQ1_to_GS05DX5H = Flux.Chain(d1, lattice[end][6:end]...)

    drift2 = ThinLens.Drift(lattice[end][5].len - d1.len)
    Sector5_Start_to_QKicker = Flux.Chain(lattice[end][1:4]..., drift2)

    allCells = [GS05MQ1_to_GS05DX5H, lattice..., Sector5_Start_to_QKicker]

    return ThinLens.NestedChain(allCells)  # this lattice contains sector 5 twice!!

end


function SPS_Cell_noDipoles(k1f::Float64=1.19761e-02, k1d::Float64=-1.19761e-02, k2f::Float64=0., k2d::Float64=0.;
    split::ThinLens.SplitScheme=split, steps::Int=steps, mergeDipoles::Bool=false)

    # quadrupoles
    qf = ThinLens.Quadrupole(3.085, k1f, 0; split=split, steps=steps)
    qd = ThinLens.Quadrupole(3.085, k1d, 0; split=split, steps=steps)

    # sextupoles
    sf = ThinLens.Sextupole(0.423, k2f, 0; split=split, steps=steps)
    sd = ThinLens.Sextupole(0.42, k2d, 0; split=split, steps=steps)

    # drifts
    d0 = ThinLens.Drift(0.7855)
    d1 = ThinLens.Drift(0.36)
    d2 = ThinLens.Drift(0.4)
    d3 = ThinLens.Drift(0.39)
    d4 = ThinLens.Drift(0.38)
    d5 = ThinLens.Drift(0.9242)
    d6 = ThinLens.Drift(0.9985)
    d7 = ThinLens.Drift(0.35)
    d8 = ThinLens.Drift(0.38)
    d9 = ThinLens.Drift(0.39)
    d10 = ThinLens.Drift(0.4)
    d11 = ThinLens.Drift(1.1488)

    # bending section merged into single drift
    bendA = ThinLens.Drift(4*6.26 + d2.len + d3.len + d4.len)
    bendB = ThinLens.Drift(4*6.26 + d8.len + d9.len + d10.len)
    return Flux.Chain(sf, d0, qf, d1, bendA, d5,
        sd, d6, qd, d7, bendB, d11)
end


function SPS_Lattice_noDipoles(k1f::Float64=1.19761e-02, k1d::Float64=-1.19761e-02, k2f::Float64=0., k2d::Float64=0.;
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1, mergeDipoles::Bool=false)

    if nested
        if cellsIdentical
            cell = SPS_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles)
            return ThinLens.NestedChain([cell for _ in 1:104])
        else
            return ThinLens.NestedChain([SPS_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles) for _ in 1:104])
        end
    else
        if cellsIdentical
            cell = SPS_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles)
            return ThinLens.FlatChain([cell for _ in 1:104])
        else
            return ThinLens.FlatChain([SPS_Cell_noDipoles(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles) for _ in 1:104])
        end
    end
end


function SPS_Cell(k1f::Float64=1.19761e-02, k1d::Float64=-1.19761e-02, k2f::Float64=0., k2d::Float64=0.;
    split::ThinLens.SplitScheme=split, steps::Int=steps, mergeDipoles::Bool=false)
    # bending magnets
    bendingAngle = 0.
    mba1 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mba2 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mba3 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mba4 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mbb1 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mbb2 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mbb3 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)
    mbb4 = ThinLens.RBen(6.26, bendingAngle, 0, 0; split=split, steps=steps)

    # quadrupoles
    qf = ThinLens.Quadrupole(3.085, k1f, 0; split=split, steps=steps)
    qd = ThinLens.Quadrupole(3.085, k1d, 0; split=split, steps=steps)

    # sextupoles
    sf = ThinLens.Sextupole(0.423, k2f, 0; split=split, steps=steps)
    sd = ThinLens.Sextupole(0.42, k2d, 0; split=split, steps=steps)

    # drifts
    d0 = ThinLens.Drift(0.7855)
    d1 = ThinLens.Drift(0.36)
    d2 = ThinLens.Drift(0.4)
    d3 = ThinLens.Drift(0.39)
    d4 = ThinLens.Drift(0.38)
    d5 = ThinLens.Drift(0.9242)
    d6 = ThinLens.Drift(0.9985)
    d7 = ThinLens.Drift(0.35)
    d8 = ThinLens.Drift(0.38)
    d9 = ThinLens.Drift(0.39)
    d10 = ThinLens.Drift(0.4)
    d11 = ThinLens.Drift(1.1488)

    if !mergeDipoles
        # set up beamline
        return Flux.Chain(sf, d0, qf, d1, mba1, d2, mba2, d3, mbb1, d4, mbb2, d5,
            sd, d6, qd, d7, mbb3, d8, mbb4, d9, mba3, d10, mba4, d11)
    else
        bendA = ThinLens.RBen(4*6.26 + d2.len + d3.len + d4.len, 4*bendingAngle, 0, 0; split=split, steps=steps)
        bendB = ThinLens.RBen(4*6.26 + d8.len + d9.len + d10.len, 4*bendingAngle, 0, 0; split=split, steps=steps)
        return Flux.Chain(sf, d0, qf, d1, bendA, d5,
            sd, d6, qd, d7, bendB, d11)
    end
end

function SPS_Lattice(k1f::Float64=1.19761e-02, k1d::Float64=-1.19761e-02, k2f::Float64=0., k2d::Float64=0.;
    nested::Bool=true, cellsIdentical::Bool=false, split::ThinLens.SplitScheme=ThinLens.splitO2nd, steps::Int=1, mergeDipoles::Bool=false)

    if nested
        if cellsIdentical
            cell = SPS_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles)
            return ThinLens.NestedChain([cell for _ in 1:104])
        else
            return ThinLens.NestedChain([SPS_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles) for _ in 1:104])
        end
    else
        if cellsIdentical
            cell = SPS_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles)
            return ThinLens.FlatChain([cell for _ in 1:104])
        else
            return ThinLens.FlatChain([SPS_Cell(k1f, k1d, k2f, k2d; split=split, steps=steps, mergeDipoles=mergeDipoles) for _ in 1:104])
        end
    end
end

end  # module