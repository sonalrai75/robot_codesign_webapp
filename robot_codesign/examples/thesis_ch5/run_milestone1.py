import numpy as np
from robot_codesign.dynamics.inertia import inertia_eigenvalues

rho = 2700.0
M1 = 0.09966
M2 = 0.000122

designs = {
    "nominal": dict(l1=0.3000, l2=0.3000, A1=0.0006, A2=0.0006,
                    target=(0.0067, 0.0752)),
    "intermediate": dict(l1=0.1419, l2=0.4581, A1=0.0019, A2=0.0002,
                         target=(0.0061, 0.0231)),
}

for name, d in designs.items():
    eig = inertia_eigenvalues(
        0.0, 0.0, d["l1"], d["l2"], d["A1"], d["A2"], rho, M1, M2
    )
    print(f"{name:12s}: Hmin={eig[0]:.8f} Hmax={eig[1]:.8f}  target={d['target']}")

print("\nNominal configuration sweep:")
for deg in range(-80, 81, 20):
    r = np.deg2rad(deg)
    eig = inertia_eigenvalues(r, 0.0, 0.3, 0.3, 0.0006, 0.0006, rho, M1, M2)
    print(f"{deg:>4d} deg : Hmin={eig[0]:.8f} Hmax={eig[1]:.8f}")
