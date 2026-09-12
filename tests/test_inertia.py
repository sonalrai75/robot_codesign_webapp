import numpy as np
from robot_codesign.dynamics.inertia import inertia_eigenvalues

RHO = 2700.0
M1 = 0.09966
M2 = 0.000122

def test_nominal_inertia_matches_thesis():
    eig = inertia_eigenvalues(0, 0, 0.3, 0.3, 0.0006, 0.0006, RHO, M1, M2)
    assert np.isclose(eig[0], 0.0067, atol=2e-5)
    assert np.isclose(eig[1], 0.0752, atol=2e-5)

def test_intermediate_inertia_matches_thesis():
    eig = inertia_eigenvalues(0, 0, 0.1419, 0.4581, 0.0019, 0.0002, RHO, M1, M2)
    assert np.isclose(eig[0], 0.0061, atol=2e-5)
    assert np.isclose(eig[1], 0.0231, atol=2e-5)

def test_worst_case_maximum_at_zero_angle():
    angles = np.deg2rad(np.linspace(-90, 90, 181))
    hmax = []
    for a in angles:
        eig = inertia_eigenvalues(a, 0, 0.3, 0.3, 0.0006, 0.0006, RHO, M1, M2)
        hmax.append(eig[1])
    assert int(np.argmax(hmax)) == 90
