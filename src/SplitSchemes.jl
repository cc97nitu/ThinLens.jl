"""Drift and kick coefficients for a symplectic drift-kick scheme."""
struct SplitScheme
    c::Vector{Float64}
    d::Vector{Float64}
end

o2nd = SplitScheme([1/2, 1/2], [1, 0])

# 3-rd order symplectic integrator by Ruth
splitO3rd = SplitScheme([1, -2/3, 2/3], [-1/24, 3/4, 7/24])

# 4th order symplectic integrator by Neri
cA = 1. / (2. * (2. - 2.0^(1/3)))
cB = (1. - 2.0^(1/3)) / (2. * (2. - 2.0^(1/3)))

dA = 1. / (2. - 2.0^(1/3))
dB = -2.0^(1/3) / (2. - 2.0^(1/3))
splitO4th = SplitScheme([cA, cB, cB, cA], [dA, dB, dA, 0.])