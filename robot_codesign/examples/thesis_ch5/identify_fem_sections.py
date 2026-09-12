import numpy as np
from scipy.optimize import least_squares
from robot_codesign.fem.two_link import flexible_frequencies_hz

E = 2.7e10
RHO = 2700.0
M1 = 0.09966
M2 = 0.000122

NOMINAL = dict(l1=0.3000, l2=0.3000, A1=0.0006, A2=0.0006)
INTERMEDIATE = dict(l1=0.1419, l2=0.4581, A1=0.0019, A2=0.0002)

TARGET_NOMINAL = np.array([116.3, 268.2])
TARGET_INTERMEDIATE = np.array([48.0, 248.0])

def first_two(design, w1, w2):
    f = flexible_frequencies_hz(
        **design, width1=w1, width2=w2, E=E, rho=RHO,
        M_joint2=M1, M_tip=M2, elements_per_link=10
    )
    return f[:2]

def residual(log_widths):
    widths = np.exp(log_widths)
    return np.log(first_two(NOMINAL, *widths) / TARGET_NOMINAL)

fit = least_squares(
    residual,
    x0=np.log([0.06, 0.06]),
    bounds=(np.log([0.001, 0.001]), np.log([0.20, 0.20])),
    xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=2000
)

w1, w2 = np.exp(fit.x)
t1, t2 = NOMINAL["A1"]/w1, NOMINAL["A2"]/w2
f_nom = first_two(NOMINAL, w1, w2)

# Two plausible carry-forward assumptions for the intermediate design:
# A) keep out-of-plane widths unchanged;
f_int_fixed_width = first_two(INTERMEDIATE, w1, w2)

# B) keep each rectangular aspect ratio unchanged.
r1 = w1/t1
r2 = w2/t2
wi1 = np.sqrt(INTERMEDIATE["A1"]*r1)
wi2 = np.sqrt(INTERMEDIATE["A2"]*r2)
f_int_fixed_aspect = first_two(INTERMEDIATE, wi1, wi2)

print("Concentrated masses INCLUDED:")
print(f"  joint-2 mass M1 = {M1:.6f} kg")
print(f"  tip mass    M2 = {M2:.6f} kg")
print()
print("Nominal inverse identification:")
print(f"  width1={w1:.8f} m, thickness1={t1:.8f} m")
print(f"  width2={w2:.8f} m, thickness2={t2:.8f} m")
print(f"  frequencies = {f_nom[0]:.6f}, {f_nom[1]:.6f} Hz")
print(f"  targets     = {TARGET_NOMINAL[0]:.6f}, {TARGET_NOMINAL[1]:.6f} Hz")
print()
print("Intermediate check, same out-of-plane widths:")
print(f"  predicted   = {f_int_fixed_width[0]:.6f}, {f_int_fixed_width[1]:.6f} Hz")
print(f"  thesis      = {TARGET_INTERMEDIATE[0]:.6f}, {TARGET_INTERMEDIATE[1]:.6f} Hz")
print()
print("Intermediate check, same aspect ratios:")
print(f"  predicted   = {f_int_fixed_aspect[0]:.6f}, {f_int_fixed_aspect[1]:.6f} Hz")
print(f"  thesis      = {TARGET_INTERMEDIATE[0]:.6f}, {TARGET_INTERMEDIATE[1]:.6f} Hz")
