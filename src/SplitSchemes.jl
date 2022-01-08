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

# 8th order symplectic integrator by Yoshida
w1 = 0.102799849391985E0
w2 = -0.196061023297549E1
w3 = 0.193813913762276E1
w4 = -0.158240635368243E0
w5 = -0.144485223686048E1
w6 = 0.253693336566229E0
w7 = 0.914844246229740E0
w0 = 1 - 2 * (w1 + w2 + w3 + w4 + w5 + w6 + w7)

coeffC = [w7 / 2, (w7 + w6) / 2, (w6 + w5) / 2, (w5 + w4) / 2, (w4 + w3) / 2, (w3 + w2) / 2,
                           (w2 + w1) / 2, (w1 + w0) / 2]
coeffC = [coeffC..., reverse(coeffC)...]

coeffD = [w7, w6, w5, w4, w3, w2, w1]
coeffD = [coeffD..., w0, reverse(coeffD)..., 0.]
const splitO8th = SplitScheme(coeffC, coeffD)