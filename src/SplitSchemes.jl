"""Drift and kick coefficients for a symplectic drift-kick scheme."""
struct SplitScheme
    c::Vector{Float64}
    d::Vector{Float64}
end

const splitO2nd = SplitScheme([1/2, 1/2], [1, 0])

# 3-rd order symplectic integrator by Ruth
const splitO3rd = SplitScheme([1, -2/3, 2/3], [-1/24, 3/4, 7/24])

# 4th order symplectic integrator by Neri
cA = 1. / (2. * (2. - 2.0^(1/3)))
cB = (1. - 2.0^(1/3)) / (2. * (2. - 2.0^(1/3)))

dA = 1. / (2. - 2.0^(1/3))
dB = -2.0^(1/3) / (2. - 2.0^(1/3))
const splitO4th = SplitScheme([cA, cB, cB, cA], [dA, dB, dA, 0.])

# 6th order symplectic integrator by Yoshida
w1 = -0.213228522200144E1
w2 = 0.426068187079180E-2
w3 = 0.143984816797678E1
w0 = 1 - 2 * (w1 + w2 + w3)

coeffC = [w3 / 2, (w3 + w2) / 2, (w2 + w1) / 2, (w1 + w0) / 2, (w1 + w0) / 2, (w2 + w1) / 2,
               (w3 + w2) / 2, w3 / 2]
coeffD = [w3, w2, w1, w0, w1, w2, w3, 0]
const splitO6th = SplitScheme(coeffC, coeffD)